# OnDeviceInfer：Qwen3 端侧大模型推理框架

面向单张消费级 NVIDIA GPU 的大模型推理实验项目。项目以 RTX 3090 24GB 和 Qwen3-14B 为主要目标，围绕显存容量与 PCIe 传输瓶颈，实现权重 Offloading、CUDA 异步 Prefetch、INT8 Weight-only 以及 INT8 KV Cache Offloading。

> 本仓库只包含源码、测试和基准脚本，不包含模型权重、量化后 Checkpoint、实验报告或 Nsight 原始数据。

## 核心功能

- Qwen3 Dense Decoder 推理，包括 GQA、Q/K Norm、RoPE、RMSNorm 与 SwiGLU；
- Safetensors 权重加载、Prefill、逐 Token Decode 和 Greedy Generation；
- FP16 权重同步 Offloading；
- 基于 Pinned Memory、独立 CUDA Stream 和双 Slot 的异步权重 Prefetch；
- 按输出通道对称 INT8 Weight-only 离线量化与 Resident/Sync/Prefetch 运行时；
- KV Cache Resident、同步 Offloading 和双 Slot Prefetch；
- HQQ-compatible Group-wise 8-bit KV Cache 压缩；
- NVTX、CUDA Event、Nsight Systems 和 SQLite 性能分析工具；
- 单元测试、正确性验证、质量评测和多种基准脚本。

## INT8 KV Cache 设计

KV Cache 使用动态分组仿射 8-bit 编码：

```text
GPU 新 K/V FP16
  → 按 Token、KV Head、64 元素分组量化
  → UINT8 data + FP16 scale/zero
  → D2H 到 CPU Pinned Memory

CPU 压缩 KV 历史
  → 下一层异步 H2D
  → GPU 反量化到复用 FP16 Slot
  → FP16 Attention
```

Qwen3-14B 的 head dimension 为 128，因此每个 Token、每个 KV Head 被分成两个 64 元素 Group。每 64 个元素由原来的 128 Bytes FP16 变为：

```text
64 Bytes UINT8 + 2 Bytes scale + 2 Bytes zero = 68 Bytes
```

理论及实测存储比例均为 53.125%，即减少 46.875%。当前实现压缩全部 Offloaded KV，不是 Hot/Cold 分页 KV；Attention 前仍会恢复为 FP16。

## RTX 3090 实测结果

环境：RTX 3090 24GB、Python 3.11、PyTorch 2.9.0+cu128、CUDA 12.8、batch size 1。

| 实验 | FP16 KV | INT8 KV | 结果 |
| --- | ---: | ---: | ---: |
| 4K+128 Host KV capacity | 660.0 MiB | 350.6 MiB | -46.875% |
| 4K+128 Prefetch Decode，3 次中位数 | 5.280 tok/s | 5.179 tok/s | 保持 98.1% |
| 16K+32 Prefetch Decode，3 次中位数 | 4.012 tok/s | 4.357 tok/s | +8.6% |
| Nsight 4K 单步 H2D | 671.10 MB / 55.18 ms | 356.53 MB / 29.85 ms | Payload -46.9% |

中文、代码和数学推理三类 512+32 Prompt 的 Teacher-forced Top-1 与 Greedy Token 一致率均为 100%。这些数据用于展示当前硬件和测试配置下的工程效果，不代表标准数据集精度或其他 GPU 上的性能。

## 项目结构

```text
ondeviceinfer/
  attention.py          # GQA、RoPE、Q/K Norm、SDPA
  cache.py              # Resident KV Cache
  kv_offload.py         # KV Sync/Prefetch Offloading
  kv_quantization.py    # FP16/HQQ-compatible INT8 KV Codec
  model.py              # FP16 Qwen3 模型
  quantization.py       # INT8 Weight-only 格式
  quantized_model.py    # INT8 权重模型
  runtime/              # Resident/Sync/Prefetch 权重运行时

benchmarks/
  quantize_checkpoint.py          # 离线量化权重
  benchmark_kv_offload.py         # KV 五组性能对照
  benchmark_kv_quantization.py    # KV Codec 微基准
  validate_kv_int8_quality.py     # Logits/KL/PPL/Token 质量验证
  profile_nsight_workload.py      # 有界 Nsight 采集入口
  analyze_nsight_sqlite.py        # Copy/Kernel 重叠分析

tests/                    # 单元测试与 CUDA 生命周期测试
requirements-lock.txt     # 当前实验环境依赖版本
```

## 环境准备

推荐使用独立 Conda 环境：

```bash
conda create -n ondeviceinfer python=3.11 -y
conda activate ondeviceinfer

pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install \
  transformers==4.57.6 \
  safetensors==0.8.0 \
  huggingface_hub==0.36.2 \
  hqq==0.2.8.post1 \
  accelerate==1.14.0 \
  pytest==9.1.1
```

如需完全复现实验服务器环境，可参考 `requirements-lock.txt`。CUDA 和 PyTorch 版本应根据本机驱动调整，不建议在不同平台上机械安装全部 NVIDIA wheel。

进入项目目录后设置模块路径：

```bash
export PYTHONPATH="$PWD"
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## 模型准备

首先将 Qwen3-14B Checkpoint 下载到本机或服务器。模型文件不会保存在 Git 仓库中。

生成项目使用的 INT8 Weight-only Checkpoint：

```bash
python benchmarks/quantize_checkpoint.py \
  --model /path/to/Qwen3-14B \
  --output /path/to/Qwen3-14B-int8
```

脚本会按层处理权重，避免同时创建完整 FP32 模型副本。量化输出目录同样不应提交到 Git。

## 运行测试

```bash
pytest -q
```

当前版本在实验服务器上的结果为：

```text
34 passed
```

部分 CUDA 测试在没有 NVIDIA GPU 时会自动跳过。

## KV Cache 性能实验

以下命令比较 Resident、FP16 Sync、INT8 Sync、FP16 Prefetch 和 INT8 Prefetch：

```bash
python benchmarks/benchmark_kv_offload.py \
  --model /path/to/Qwen3-14B-int8 \
  --runs resident,sync:64,prefetch:64 \
  --storage-formats fp16,int8_hqq \
  --kv-group-size 64 \
  --hqq-runtime-backend torch \
  --input-tokens 4096 \
  --max-new-tokens 128 \
  --window-tokens 32 \
  --repeats 3 \
  --output outputs/kv_int8_4k128.json
```

`hqq_runtime_backend=torch` 使用与 HQQ `optimize=False` 数值一致、面向在线缓存的预分配实现；`hqq_reference` 用于数值对照，不建议作为逐层热路径。

## KV Cache 质量验证

```bash
python benchmarks/validate_kv_int8_quality.py \
  --model /path/to/Qwen3-14B-int8 \
  --input-tokens 512 \
  --max-new-tokens 32 \
  --kv-budget-mib 64 \
  --kv-group-size 64 \
  --output outputs/kv_int8_quality.json
```

输出包含：

- K/V 重建误差；
- Logits Cosine、MSE 与 KL Divergence；
- Teacher-forced Top-1 一致率；
- Greedy Token 一致率；
- Reference Token Stream 上的困惑度变化；
- Cache 长度、显存和有限值检查。

## Nsight Systems 分析

```bash
ONDEVICEINFER_NVTX=1 nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=outputs/kv_int8_profile \
  python benchmarks/profile_nsight_workload.py \
    --model /path/to/Qwen3-14B-int8 \
    --scenario kv-prefetch \
    --input-tokens 4096 \
    --capture-steps 1 \
    --kv-budget-mib 64 \
    --kv-storage-format int8_hqq \
    --output outputs/kv_int8_profile.json
```

导出 SQLite 后可计算真实时间轴上的 Copy/Kernel 重叠：

```bash
nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output=outputs/kv_int8_profile.sqlite \
  outputs/kv_int8_profile.nsys-rep

python benchmarks/analyze_nsight_sqlite.py \
  outputs/kv_int8_profile.sqlite \
  --output outputs/kv_int8_overlap.json
```

分析时应区分多个 Stream 的操作时长之和与合并后的时间轴占用，不能直接把 H2D、量化、反量化和 Attention 的时间相加作为端到端延迟。

## 当前限制

- 当前主要面向 batch size 1 和单卡推理；
- INT8 权重仍在计算前物化为 FP16，没有接入融合 INT8/INT4 GEMM；
- INT8 KV 在 Attention 前恢复为 FP16，没有融合到 Attention Kernel；
- 还未实现 Hot FP16 + Cold INT8 的分页状态机；
- 质量实验规模有限，需要标准数据集和更长生成进一步验证；
- 代码以研究、学习和工程实验为目标，尚未封装稳定的服务端 API。

## 后续方向

- AWQ/Marlin W4A16 融合权重计算；
- Triton/CUDA 融合 KV quantize/dequantize；
- GPU-first Paged KV 与 Hot/Cold 分层；
- 根据上下文长度自动选择 FP16 或 INT8 KV；
- CUDA Graph、算子融合和更完整的精度评测。
