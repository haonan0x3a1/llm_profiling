# nys_llm.py 使用说明

> 这是我自用的 LLM 推理性能分析脚本，集成了 vLLM 和 Nsight Systems，用于快速测试和调优大型语言模型的推理性能

## 🤖 基本功能

* 测试各种开源大模型（Qwen、DeepSeek、LLaMA等）的推理性能
* 测量吞吐量（tokens/秒）、延迟（ms/token）等关键指标
* 支持 Nsight Systems 进行 GPU 深度分析
* 自动保存所有配置和结果文件

## 🚀 怎么用

### 最简单方式（交互式）

```
python nys_llm.py
```

脚本会一步步引导你：

1. 从模型列表中选择要测试的模型
2. 输入要使用的 GPU 数量
3. 自动跑测试并生成结果

### 高级用法（直接指定参数）

```
# 普通性能测试 (Qwen2.5-72B, 8张GPU)
python nys_llm.py --model Qwen2.5-72B-Instruct --gpus 8

# 带Nsys GPU分析 (结果会更详细)
python nys_llm.py --nsys --model Qwen2.5-72B-Instruct --gpus 8

# 自定义输出目录
python nys_llm.py --model Qwen2.5-72B-Instruct --gpus 8 --result-dir /path/to/my_test
```

## 📂 结果文件说明

测试完成后，会在 `results/` 目录下生成一个时间戳命名的文件夹，里面包含：

### 📝 核心文件

* **config.txt** - 测试配置参数（用了什么模型、GPU数量等）
* **metrics.txt** - 性能数据（吞吐量、延迟等关键指标）
* **generated_output.txt** - 模型实际生成的文本示例

### 🔍 Nsys分析文件（如果用了 `--nsys` 参数）

* **vllm_profile.nsys-rep** - Nsight Systems 报告文件
  * 用 `nsight-sys vllm_profile.nsys-rep` 打开
  * 可以看到GPU活动细节和性能瓶颈
* **nsys.log** - Nsys运行的日志
* **vllm_profile.sqlite** - 底层SQLite数据库（一般不用看）

### 📊 metrics.txt 示例解读

```
total_requests: 1              ← 跑了1次推理
total_time: 3.739               ← 总耗时3.739秒
total_tokens: 149               ← 生成了149个token
avg_time_per_request: 3.739     ← 平均每次推理3.739秒
throughput: 39.85               ← 吞吐量：39.85 tokens/秒
latency_per_token: 25.09        ← 延迟：每个token耗时25.09毫秒
```

## 🛠 我的自用技巧

### 修改默认设置

编辑脚本开头的常量:

```
# 模型存放位置
MODELS_BASE_DIR = "/home/wanghaonan/project/llm_profiling/models/"

# 显存利用率（如果出OOM错误就调低它）
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85

# 生成token数量（测试长文本时加大这个值）
DEFAULT_MAX_NEW_TOKENS = 256

# 测试次数（多次测试取平均）
NUM_REQUESTS_TO_PROFILE = 1
```

### Nsys分析常用命令

```
# 打开图形界面
nsight-sys 结果目录/vllm_profile.nsys-rep

# 生成命令行报告
nsys stats --report gputrace 结果目录/vllm_profile.nsys-rep
```

### 批量测试脚本

```
#!/bin/bash
# run_all.sh

models=("Qwen2.5-72B-Instruct" "DeepSeek-R1-Distill-Llama-70B")
gpus=8

for model in "${models[@]}"; do
    # 普通性能测试
    python nys_llm.py --model $model --gpus $gpus
  
    # Nsys深度分析
    python nys_llm.py --nsys --model $model --gpus $gpus
  
    # 休息1分钟让GPU凉快下
    sleep 60
done
```

运行：`chmod +x run_all.sh && ./run_all.sh`

## 🐞 常见问题

### 运行中崩溃怎么办？

1. 先检查 `nsys.log` 或终端输出的错误信息
2. 如果报显存不足：
   * 降低 `DEFAULT_GPU_MEMORY_UTILIZATION` 到 0.8
   * 减少 `NUM_REQUESTS_TO_PROFILE` (默认是1)
3. 如果Nsys报错：
   * 尝试简化命令：`nsys profile -t cuda python nys_llm.py ...`

### 模型路径不对？

在脚本开头修改 `MODELS_BASE_DIR` 指向你实际存放模型的目录

### 结果文件乱？

清理旧结果：`rm -rf results/*`（注意别删错）
