# AWQ W4A16 / Marlin 复现指南

本次发布只包含代码、测试和使用说明。原始/量化权重、校准文本、参考张量、生成结果、实验报告、GPU环境和分析捕获文件都留在本地。

## 实现与适用范围

- LLM Compressor AWQ缩放搜索，`n_grid=20`、`duo_scaling=True`；没有额外配置Clipping搜索，也不追加GPTQ。
- 40层、每层7个Linear，共280个目标；INT4仿射分组、G=128、FP16激活。Embedding/LM Head/Norm保持高精度，AWQ变换后的Norm参数一并保存加载。
- CPU初始加载、按Qwen3DecoderLayer逐层校准，校准Batch=1。通过FP32损失计算避免FP16平方误差求和溢出，并适配库重放Linear时的关键字/位置参数差异。
- 库按适用映射执行等价缩放；GQA下V→O维度不匹配映射会跳过，不能宣称全部280个Linear都执行了同一种缩放。
- 启动时先上传压缩权重，再在GPU Repack和排列元数据。Decoder压缩权重常驻，GPU LM Head和最终Norm常驻，Embedding在CPU。
- 调用vLLM的现成Marlin CUDA内核；模型前向、生成与缓存管理由项目执行。不是修改vLLM LLMEngine，也不是自行编写Marlin内核。
- 本发布主线为Resident FP16 KV；AWQ与KV Offloading联动性能没有用这里的Resident结果代替。

## 两个独立环境

已验证目标：Linux、Python3.11、RTX3090/SM86、CUDA12.8、Torch2.9.0+cu128。完整CUDA推理需要NVIDIA GPU；CPU测试不代表GPU数值验证通过。

从仓库根目录执行：

```bash
conda create -n ondeviceinfer-awq-cal python=3.11 -y
conda activate ondeviceinfer-awq-cal
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-awq-calibration.txt
python -m pip check

conda create -n ondeviceinfer-awq-run python=3.11 -y
conda activate ondeviceinfer-awq-run
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-awq-inference.txt
python -m pip check
```

这些文件锁定主要直接依赖，不是所有平台的完整传递依赖锁。不要将`requirements-lock.txt`再整体安装到这两个环境：原基线环境的版本不完全相同。

校准使用compressed-tensors0.14.0，推理使用0.12.2，因此分开环境。在本项目SM86真实权重测试中，旧vLLM0.11.2存在数值稳定性问题，适配层默认拒绝该版本；正式使用0.12.0。不要为了运行而解除这个检查。

## 准备校准与评测数据

先自行下载完整的Qwen3-14B模型。下列路径均为占位符，模型和生成文件不要放入Git。

```bash
export PYTHONPATH="$PWD"
export FP16_MODEL=/path/to/Qwen3-14B
export AWQ_MODEL=/path/to/Qwen3-14B-awq

conda activate ondeviceinfer-awq-cal
python benchmarks/prepare_weight_quant_data.py \
  --model "$FP16_MODEL" \
  --cache /path/to/local-dataset-cache \
  --output outputs/awq/dataset.json
```

脚本从固定版本BELLE/Dolly/MBPP获取数据，固定seed42并去重；校准/开发/测试源样本不相交，数量128/32/96。校准中文48、英文48、代码32，每条最多512 Tokens；本机冻结数据有效校准Tokens为16,674。数据和Manifest只写入忽略的输出目录。

## 分层AWQ校准

```bash
python benchmarks/calibrate_weight_quant.py \
  --model "$FP16_MODEL" \
  --algorithm awq \
  --data outputs/awq/dataset.json \
  --samples 128 --length 512 \
  --output "$AWQ_MODEL" \
  --report outputs/awq/calibration.json
```

输出compressed-tensors Safetensors分片、Tokenizer及实验Manifest。已有完成的checkpoint不会被覆盖。不得在校准后将原始Norm权重直接混入AWQ模型；这样可能破坏折叠缩放的等价关系。

## 算子与部署性能

```bash
conda activate ondeviceinfer-awq-run
export PYTHONPATH="$PWD"
export ONDEVICEINFER_MARLIN_FP32_REDUCE=0
export ONDEVICEINFER_MARLIN_MAX_ROWS=0

python benchmarks/validate_weight_quant_kernels.py \
  --model "$AWQ_MODEL" --backend marlin \
  --output outputs/awq/kernel_golden.json

python benchmarks/benchmark_weight_quant_models.py \
  --model "$AWQ_MODEL" --backend marlin \
  --contexts 512,4096 --batches 1,2 \
  --generation 128 --repeats 3 \
  --output outputs/awq/performance.json
```

Golden使用实际首/中/末层7类Linear、M=1/4/32/256，共84个矩阵/输入行数组合，与同一量化权重的FP16解码参考比较。它验证后端适配，不等同于相对原始FP16模型的任务质量。

`MAX_ROWS=0`为完整输入、不额外分行。`FP32_REDUCE=0`控制跨CTA归约路径，不表示整个模型都不使用FP32累加。小样本/完整模型重复中仍可有微小浮点波动，不能保证逐位确定性。

Batch>1报告总tokens/s；128输出包含Prefill得到的首Token，Decode计127步。加载与Repack不计入TTFT。量化收益与融合后端收益需要分别解释。

## 三路模型质量评测

A为原始FP16，B为同一AWQ checkpoint的显式QDQ+FP16 Linear，C为AWQ Marlin。全部使用Resident FP16 KV。

```bash
# A：FP16参考。32为GPU Decoder Layer预算，需要根据可用显存检查。
python benchmarks/validate_weight_quant.py \
  --model "$FP16_MODEL" --backend fp16 --fp16-budget 32 \
  --data outputs/awq/dataset.json \
  --reference-dir outputs/awq/fp16_reference \
  --output outputs/awq/quality_fp16.json

# B：量化影响，省去慢速QDQ自由生成。
python benchmarks/validate_weight_quant.py \
  --model "$AWQ_MODEL" --backend reference_int4 --skip-greedy \
  --data outputs/awq/dataset.json \
  --reference-dir outputs/awq/fp16_reference \
  --reference-meta outputs/awq/quality_fp16.json \
  --output outputs/awq/quality_qdq.json

# C：实际部署模型，包含96条固定128 Token自由生成。
python benchmarks/validate_weight_quant.py \
  --model "$AWQ_MODEL" --backend marlin \
  --data outputs/awq/dataset.json \
  --reference-dir outputs/awq/fp16_reference \
  --reference-meta outputs/awq/quality_fp16.json \
  --output outputs/awq/quality_marlin.json
```

FP16参考需要权重Offloading，因此完整自由生成可能耗时较长；输出会逐样本/分组保存，可用相同命令续跑。不能更换Tokenizer、数据、前向代码或配置后继续混用旧参考。

每种语言的独立源文本65,536输入Tokens、非重叠窗口1024，有效目标65,472；不是WikiText/C4。Teacher取96条样本最后16个真实位置，包含模板/结束符；自由生成固定128 Tokens、不提前EOS，按相同Prompt长度分组B≤4，无Padding。

`--skip-greedy`表示只完成PPL/Teacher；其JSON的`complete=true`仅代表所选阶段完成。完整评测还必须检查A/C各96条`generated_ids`均为128，以及B/C每条40项`layer_output_errors`。

三路PPL与模型输出对比用于区分量化影响和后端漂移，误差不能直接相减。PPL增幅不是答题准确率下降，Top1一致率不是任务准确率，自由生成分叉也不等同于回答错误。简历中的质量结论应限定数据与协议。

## CPU回归

安装基础测试依赖后，从仓库根运行：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

测试覆盖INT4位序、分组Scale/Zero-point、抽样解码、融合接口形状、FP16参考协议检查及原有功能。没有GPU时CUDA生命周期测试跳过。不得将CPU测试通过表述为真实GPU算子性能测试通过。
