#!/usr/bin/env python3
"""Convert a Qwen3 checkpoint to the project's auditable INT8 format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from ondeviceinfer.config import Qwen3Config
from ondeviceinfer.quantization import (
    QUANTIZATION_FORMAT,
    QUANTIZATION_VERSION,
    quantize_linear_weight,
)
from ondeviceinfer.runtime.offload import LAYER_TENSOR_NAMES, tensor_bytes
from ondeviceinfer.runtime.quantized import (
    LINEAR_WEIGHT_NAMES,
    NORM_WEIGHT_NAMES,
    qweight_name,
    scale_name,
)
from ondeviceinfer.weights import SafetensorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--row-block-size", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def error_metrics(
    original: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    *,
    row_block_size: int,
) -> dict[str, float]:
    absolute_sum = 0.0
    squared_sum = 0.0
    reference_squared_sum = 0.0
    maximum = 0.0
    elements = original.numel()
    for start in range(0, original.shape[0], row_block_size):
        end = min(start + row_block_size, original.shape[0])
        reference = original[start:end].float()
        reconstructed = qweight[start:end].float() * scale[start:end, None].float()
        difference = reconstructed - reference
        absolute_sum += difference.abs().sum().item()
        squared_sum += difference.square().sum().item()
        reference_squared_sum += reference.square().sum().item()
        maximum = max(maximum, difference.abs().max().item())
    return {
        "mae": absolute_sum / elements,
        "rmse": (squared_sum / elements) ** 0.5,
        "max_abs_error": maximum,
        "relative_l2": (squared_sum / max(reference_squared_sum, 1e-30)) ** 0.5,
    }


def copy_metadata(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix in {".safetensors", ".bin"}:
            continue
        if path.name == "model.safetensors.index.json":
            continue
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(path.name)
    return copied


def main() -> None:
    args = parse_args()
    source = args.model.resolve()
    output = args.output
    if args.row_block_size <= 0:
        raise ValueError("row-block-size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory {output}"
        )

    started = time.time()
    config = Qwen3Config.from_json(source / "config.json")
    index = SafetensorIndex(source)
    revision = args.revision or source.name
    quantization_config = {
        "format": "qwen3-int8-weight-only-v1",
        "algorithm": QUANTIZATION_FORMAT,
        "algorithm_version": QUANTIZATION_VERSION,
        "axis": 0,
        "zero_point": 0,
        "quant_min": -127,
        "quant_max": 127,
        "scale_dtype": "float16",
        "compute_dtype": "float16",
        "row_block_size": args.row_block_size,
        "quantized_tensors": "decoder_linear_weights_only",
        "unquantized_tensors": [
            "embedding",
            "rms_norms",
            "q_norm",
            "k_norm",
            "lm_head",
        ],
        "source_model": str(source),
        "source_revision": revision,
    }
    (output / "quantization_config.json").write_text(
        json.dumps(quantization_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_files = copy_metadata(source, output)

    error_path = output / "tensor_quant_error.jsonl"
    error_rows: list[dict[str, object]] = []
    original_decoder_bytes = 0
    quantized_decoder_bytes = 0
    tensor_count = 0

    with error_path.open("w", encoding="utf-8") as error_file:
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}."
            saved: dict[str, torch.Tensor] = {}
            for suffix in LAYER_TENSOR_NAMES:
                source_tensor = index.get_tensor(prefix + suffix)
                original_decoder_bytes += tensor_bytes(source_tensor)
                if suffix in NORM_WEIGHT_NAMES:
                    saved[suffix] = source_tensor.to(torch.float16).contiguous()
                    continue
                if suffix not in LINEAR_WEIGHT_NAMES:
                    raise RuntimeError(f"Unclassified layer tensor {suffix}")
                quantized = quantize_linear_weight(
                    source_tensor,
                    row_block_size=args.row_block_size,
                    scale_dtype=torch.float16,
                )
                saved[qweight_name(suffix)] = quantized.qweight
                saved[scale_name(suffix)] = quantized.scale
                metrics = error_metrics(
                    source_tensor,
                    quantized.qweight,
                    quantized.scale,
                    row_block_size=args.row_block_size,
                )
                row: dict[str, object] = {
                    "layer_idx": layer_idx,
                    "tensor": prefix + suffix,
                    "shape": list(source_tensor.shape),
                    "source_dtype": str(source_tensor.dtype).removeprefix("torch."),
                    "qweight_dtype": "int8",
                    "scale_dtype": "float16",
                    "source_bytes": tensor_bytes(source_tensor),
                    "quantized_bytes": quantized.storage_nbytes,
                    "compression_ratio": quantized.storage_nbytes
                    / tensor_bytes(source_tensor),
                    **metrics,
                }
                error_rows.append(row)
                error_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                error_file.flush()
                tensor_count += 1

            target = output / f"layer-{layer_idx:03d}.safetensors"
            temporary = output / f"layer-{layer_idx:03d}.safetensors.tmp"
            save_file(saved, temporary)
            os.replace(temporary, target)
            quantized_decoder_bytes += sum(tensor_bytes(tensor) for tensor in saved.values())
            print(
                f"quantized layer {layer_idx + 1}/{config.num_hidden_layers}: "
                f"{target.stat().st_size / 2**20:.2f} MiB",
                flush=True,
            )
            del saved

    globals_to_save = {
        "model.embed_tokens.weight": index.get_tensor(
            "model.embed_tokens.weight"
        ).to(torch.float16).contiguous(),
        "model.norm.weight": index.get_tensor("model.norm.weight")
        .to(torch.float16)
        .contiguous(),
        "lm_head.weight": index.get_tensor("lm_head.weight")
        .to(torch.float16)
        .contiguous(),
    }
    global_path = output / "global.safetensors"
    global_temporary = output / "global.safetensors.tmp"
    save_file(globals_to_save, global_temporary)
    os.replace(global_temporary, global_path)
    global_bytes = sum(tensor_bytes(tensor) for tensor in globals_to_save.values())
    del globals_to_save

    tensor_files = sorted(output.glob("*.safetensors"))
    files = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in tensor_files
    ]
    manifest = {
        **quantization_config,
        "schema_version": 1,
        "created_unix_seconds": time.time(),
        "elapsed_seconds": time.time() - started,
        "source_config_sha256": sha256_file(source / "config.json"),
        "source_index_total_size": index.total_size,
        "num_layers": config.num_hidden_layers,
        "quantized_linear_tensor_count": tensor_count,
        "original_decoder_bytes": original_decoder_bytes,
        "quantized_decoder_bytes": quantized_decoder_bytes,
        "decoder_compression_ratio": quantized_decoder_bytes
        / original_decoder_bytes,
        "global_fp16_bytes": global_bytes,
        "checkpoint_tensor_bytes": quantized_decoder_bytes + global_bytes,
        "checkpoint_ratio_vs_source": (quantized_decoder_bytes + global_bytes)
        / index.total_size,
        "metadata_files": metadata_files,
        "tensor_files": files,
        "quantization_error_summary": {
            "max_tensor_mae": max(float(row["mae"]) for row in error_rows),
            "max_tensor_rmse": max(float(row["rmse"]) for row in error_rows),
            "max_tensor_max_abs_error": max(
                float(row["max_abs_error"]) for row in error_rows
            ),
            "max_tensor_relative_l2": max(
                float(row["relative_l2"]) for row in error_rows
            ),
        },
    }
    (output / "quantization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
