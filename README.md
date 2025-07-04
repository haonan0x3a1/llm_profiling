# LLM Profiling Framework

一个用于LLM（大语言模型）性能分析的自动化框架，支持使用NVIDIA Nsight System和Nsight Compute进行深度性能分析。

## 功能特性

- 🚀 **多种分析模式**：支持纯推理、Nsight System分析、完整性能分析
- 🔧 **灵活配置**：支持YAML配置文件和命令行参数
- 📊 **自动可视化**：生成性能图表和HTML报告
- 🎯 **批量实验**：支持参数扫描和批量分析
- 💾 **结构化输出**：实验结果自动组织到独立文件夹
- 🔍 **详细分析**：支持kernel级别的性能分析

## 目录结构

```
code/
├── profiler/           # 核心分析模块
│   ├── __init__.py
│   ├── config.py       # 配置管理
│   ├── utils.py        # 工具函数
│   ├── nsys_analyzer.py # Nsight System分析器
│   ├── ncu_analyzer.py  # Nsight Compute分析器
│   ├── visualizer.py   # 可视化模块
│   └── runner.py       # 实验运行器
├── scripts/            # 运行脚本
│   ├── run_single.py   # 单次实验
│   └── run_batch.py    # 批量实验
├── configs/            # 配置文件
│   └── default.yaml    # 默认配置
├── requirements.txt    # Python依赖
└── README.md          # 本文档
```

## 安装和环境设置

### 1. 安装Python依赖

```bash
cd /home/wanghaonan/project/llm_profiling/code
pip install -r requirements.txt
```

### 2. 安装vLLM

```bash
# 根据你的CUDA版本安装vLLM
pip install vllm
```

### 3. 安装NVIDIA工具（如果尚未安装）

- NVIDIA Nsight Systems
- NVIDIA Nsight Compute

## 使用方法

### 快速开始

#### 1. 单次实验运行

```bash
# 最简单的推理测试
python scripts/run_single.py --model Qwen2.5-72B-Instruct

# 使用Nsight System分析
python scripts/run_single.py \
    --model DeepSeek-R1-Distill-Llama-70B \
    --profile-mode nsys \
    --prompt "请解释深度学习的基本原理" \
    --max-tokens 500

# 完整分析（nsys + ncu）
python scripts/run_single.py \
    --model Qwen2.5-72B-Instruct \
    --profile-mode nsys_ncu \
    --batch-size 4 \
    --tensor-parallel-size 2
```

#### 2. 批量实验

```bash
# 使用预设参数
python scripts/run_batch.py \
    --model Qwen2.5-72B-Instruct \
    --preset quick \
    --profile-mode inference_only

# 自定义参数扫描
python scripts/run_batch.py \
    --model DeepSeek-R1-Distill-Llama-70B \
    --parameter-sweep "batch_size:1,2,4;max_tokens:100,200,500" \
    --profile-mode nsys
```

### 详细使用说明

#### 命令行参数

**单次实验 (run_single.py)**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称（必需） | - |
| `--prompt` | 输入提示 | "Hello, how are you?" |
| `--max-tokens` | 最大生成token数 | 100 |
| `--temperature` | 采样温度 | 0.7 |
| `--batch-size` | 批次大小 | 1 |
| `--tensor-parallel-size` | 张量并行大小 | 1 |
| `--profile-mode` | 分析模式 | inference_only |
| `--experiment-name` | 自定义实验名称 | 自动生成 |

**批量实验 (run_batch.py)**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型名称（必需） | Qwen2.5-72B-Instruct |
| `--base-prompt` | 基础提示 | "Hello, how are you?" |
| `--profile-mode` | 分析模式 | inference_only |
| `--parameter-sweep` | 参数扫描规格 | "batch_size:1,2,4;max_tokens:50,100" |
| `--preset` | 预设实验配置 | quick/comprehensive/scaling |

#### 分析模式详解

1. **inference_only**: 
   - 仅运行推理，记录基本性能指标
   - 最快，适合功能测试和基本性能评估

2. **nsys**: 
   - 使用Nsight System进行系统级分析
   - 提供CUDA kernel、API调用、内存使用等详细信息
   - 生成.nsys-rep和.sqlite文件

3. **nsys_ncu**: 
   - 完整分析，包含Nsight System + Nsight Compute
   - 提供kernel级别的详细性能指标
   - 最全面但也最耗时

#### 预设实验配置

- **quick**: 快速测试（batch_size: 1,2; max_tokens: 50,100）
- **comprehensive**: 全面测试（包含多种batch_size、max_tokens、temperature组合）
- **scaling**: 扩展性测试（重点测试并行性能）

### 配置文件使用

#### 1. 使用默认配置

```bash
python -c "
from profiler.config import ConfigManager, ExperimentConfig
from profiler.runner import ExperimentRunner

config = ConfigManager.load_config('configs/default.yaml')
runner = ExperimentRunner(config)
results = runner.run()
print(f'实验完成: {results[\"status\"]}')
"
```

#### 2. 自定义配置文件

创建自定义YAML配置：

```yaml
# my_experiment.yaml
model_path: "/home/wanghaonan/project/llm_profiling/models/DeepSeek-R1-Distill-Llama-70B"
model_name: "DeepSeek-R1-Distill-Llama-70B"
prompt: "请详细解释量子计算的基本原理和应用前景"
max_tokens: 1000
batch_size: 2
tensor_parallel_size: 4
profile_mode: "nsys_ncu"
```

使用自定义配置：

```python
from profiler.config import ConfigManager
from profiler.runner import ExperimentRunner

config = ConfigManager.load_config('my_experiment.yaml')
runner = ExperimentRunner(config)
results = runner.run()
```

## 结果文件说明

每次实验会在`/home/wanghaonan/project/llm_profiling/results/`下创建独立文件夹，命名格式：
`{model_name}_{profile_mode}_{timestamp}`

### 文件结构示例

```
results/
└── Qwen2.5-72B-Instruct_nsys_20241214_143052/
    ├── config.yaml              # 实验配置
    ├── experiment.log           # 详细日志
    ├── summary.json             # 实验摘要
    ├── gpu_info.json           # GPU信息
    ├── generated_output.txt     # 生成文本
    ├── metrics.json            # 基本性能指标
    ├── inference_script.py     # 执行脚本
    ├── vllm_profile.nsys-rep   # Nsight System报告
    ├── vllm_profile.sqlite     # SQLite数据库
    ├── nsys_metrics.json       # Nsight System分析结果
    ├── nsys_summary.csv        # CSV格式摘要
    ├── ncu_metrics.json        # Nsight Compute分析结果（如果适用）
    └── kernel_comparison.csv   # Kernel对比分析（如果适用）
```

### 可视化文件

可视化结果保存在`/home/wanghaonan/project/llm_profiling/results/visualization/`：

- `{experiment_name}_kernel_time.png` - Kernel时间分布图
- `{experiment_name}_cuda_api.png` - CUDA API使用分布
- `{experiment_name}_dashboard.png` - 综合仪表板
- `{experiment_name}_report.html` - HTML报告

## 高级功能

### 1. 程序化API使用

```python
from profiler.config import ExperimentConfig
from profiler.runner import ExperimentRunner, BatchRunner

# 创建配置
config = ExperimentConfig(
    model_path="/home/wanghaonan/project/llm_profiling/models/Qwen2.5-72B-Instruct",
    prompt="分析人工智能在医疗领域的应用",
    max_tokens=300,
    profile_mode="nsys"
)

# 运行单次实验
runner = ExperimentRunner(config)
results = runner.run()

# 批量实验
batch_runner = BatchRunner(config)
batch_results = batch_runner.run_parameter_sweep({
    "batch_size": [1, 2, 4],
    "max_tokens": [100, 200, 300]
})
```

### 2. 自定义可视化

```python
from profiler.visualizer import Visualizer

# 为特定实验生成可视化
viz = Visualizer("/path/to/experiment/results")
viz.create_all_visualizations()
html_report = viz.generate_report_html()
```

### 3. 结果分析

```python
import json
import pandas as pd

# 加载实验结果
with open("results/experiment/nsys_metrics.json", 'r') as f:
    metrics = json.load(f)

# 分析top kernels
top_kernels = metrics['summary']['top_kernels']
for kernel in top_kernels[:5]:
    print(f"{kernel['name']}: {kernel['time_percentage']:.2f}%")

# 加载CSV进行进一步分析
df = pd.read_csv("results/experiment/nsys_summary.csv")
print(df.describe())
```

## 常见问题

### Q: 如何选择合适的tensor_parallel_size？

A: 根据你的GPU数量和模型大小：
- 单GPU：使用1
- 多GPU：通常设置为GPU数量，如2、4、8
- 大模型（70B+）：建议使用多GPU并行

### Q: 实验失败怎么办？

A: 检查以下几点：
1. 确认模型路径存在
2. 检查GPU显存是否足够
3. 查看实验日志文件
4. 确认vLLM和CUDA环境正确安装

### Q: 如何减少分析时间？

A: 
- 使用`inference_only`模式进行快速测试
- 减少`max_tokens`和`batch_size`
- 使用`quick`预设进行批量实验

### Q: 如何分析特定的kernel？

A: 在配置文件中设置：
```yaml
ncu_options:
  kernel_name: "your_kernel_name_pattern"
```

### Q: 结果文件太大怎么办？

A: 
- `.nsys-rep`文件较大是正常的，包含详细trace信息
- 可以删除中间文件，保留`.json`和`.csv`摘要
- 使用基本模式而非完整分析模式

## 扩展开发

### 添加新的分析器

1. 在`profiler/`目录创建新的分析器模块
2. 继承基础接口并实现分析逻辑
3. 在`runner.py`中集成新分析器

### 添加新的可视化

1. 在`visualizer.py`中添加新的绘图方法
2. 更新`create_all_visualizations()`方法
3. 添加相应的配置选项

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 添加测试
4. 提交Pull Request

## 许可证

本项目用于学术研究目的。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目路径：`/home/wanghaonan/project/llm_profiling`
- 日志文件：查看各实验目录下的`experiment.log`

---

**最后更新**: 2024年12月14日 