#!/usr/bin/env python3
"""Freeze disjoint, revision-pinned calibration and evaluation token sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import requests
from transformers import AutoTokenizer

SOURCES = {
    "en": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "databricks-dolly-15k.jsonl"),
    "zh": ("BelleGroup/train_0.5M_CN", "29c35e9b56c7fc91f04a613ed33896f7a19bad54", "Belle_open_source_0.5M.json"),
    "code_train": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "sanitized/train-00000-of-00001.parquet"),
    "code_dev": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "sanitized/validation-00000-of-00001.parquet"),
    "code_test": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "sanitized/test-00000-of-00001.parquet"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(key: str, root: Path) -> Path:
    repo, revision, filename = SOURCES[key]
    path = root / (key + Path(filename).suffix)
    if path.exists():
        return path
    for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
        try:
            response = requests.get(f"{endpoint}/datasets/{repo}/resolve/{revision}/{filename}", stream=True, timeout=(20, 120))
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".partial")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(2**20):
                    handle.write(chunk)
            temporary.replace(path)
            print(f"downloaded {key}: {path.stat().st_size / 2**20:.1f} MiB", flush=True)
            return path
        except requests.RequestException as error:
            print(f"{endpoint}: {type(error).__name__}: {error}", flush=True)
    raise RuntimeError(f"failed to download {key}")


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    # BELLE is line-delimited JSON despite the .json filename.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/mnt/ondeviceinfer/models/Qwen3-14B"))
    parser.add_argument("--cache", type=Path, default=Path("/mnt/ondeviceinfer/datasets/weight_quant_compare"))
    parser.add_argument("--output", type=Path, default=Path("reports/raw/weight_quant_compare/dataset.json"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rng = random.Random(42)
    splits = {"calibration": [], "development": [], "test": []}
    corpora = {"zh": [], "en": []}
    seen: set[str] = set()
    manifest = {}
    for language in ("zh", "en", "code"):
        keys = [language] if language != "code" else ["code_train", "code_dev", "code_test"]
        queues = {}
        for key in keys:
            path = download(key, args.cache)
            manifest[key] = {"repository": SOURCES[key][0], "revision": SOURCES[key][1], "filename": SOURCES[key][2], "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
            rows = list(enumerate(read_rows(path)))
            rng.shuffle(rows)
            queues[key] = iter(rows)
        for split, count in (("calibration", 32 if language == "code" else 48), ("development", 8 if language == "code" else 12), ("test", 24 if language == "code" else 36)):
            key = language if language != "code" else {"calibration": "code_train", "development": "code_dev", "test": "code_test"}[split]
            for _ in range(count):
                while True:
                    index, row = next(queues[key])
                    prompt = row.get("instruction", row.get("prompt", ""))
                    answer = row.get("output", row.get("response", ""))
                    if language == "en":
                        context = row.get("context", "")
                        if context:
                            prompt += "\n" + context
                    if language == "code":
                        prompt, answer = row["prompt"], row["code"]
                    text_hash = digest((prompt.strip() + "\n" + answer.strip()).encode())
                    if not prompt or not answer or text_hash in seen:
                        continue
                    seen.add(text_hash)
                    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
                    ids = tokenizer(text, add_special_tokens=False).input_ids[:512]
                    prompt_ids = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True, enable_thinking=False)[:512]
                    if len(ids) < 16:
                        continue
                    splits[split].append({"source": key, "source_row": index, "language": language, "text_sha256": text_hash, "input_ids": ids, "prompt_ids": prompt_ids})
                    break
        if language in corpora:
            # Separate held-out source rows, not reference-generated continuations.
            while len(corpora[language]) < 65536:
                index, row = next(queues[language])
                prompt = row.get("instruction", "")
                answer = row.get("output", row.get("response", ""))
                text_hash = digest((prompt.strip() + "\n" + answer.strip()).encode())
                if text_hash in seen:
                    continue
                seen.add(text_hash)
                text = prompt + "\n" + answer + "\n\n"
                corpora[language].extend(tokenizer(text, add_special_tokens=False).input_ids)
            corpora[language] = corpora[language][:65536]
    result = {"schema_version": 1, "seed": 42, "tokenizer_model": str(args.model.resolve()), "thinking": False, "sources": manifest, "splits": splits, "corpora": corpora, "corpus_label": "held-out BELLE/Dolly source text; not WikiText/C4", "split_effective_tokens": {key: sum(len(row['input_ids']) for row in rows) for key, rows in splits.items()}}
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    manifest_path = args.output.with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps({key: value for key, value in result.items() if key not in ("splits", "corpora")} | {"dataset_sha256": digest(serialized.encode()), "split_counts": {key: len(rows) for key, rows in splits.items()}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["split_effective_tokens"]), flush=True)


if __name__ == "__main__":
    main()
