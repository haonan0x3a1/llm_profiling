#!/usr/bin/env python3
"""
批量实验运行脚本
"""
import sys
import os
import argparse
import json
from pathlib import Path

# 添加父目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.config import ExperimentConfig
from profiler.runner import BatchRunner


def parse_parameter_ranges(param_str):
    """解析参数范围字符串"""
    # 格式: "batch_size:1,2,4;max_tokens:50,100,200"
    if not param_str:
        return {}
    
    parameters = {}
    for param_spec in param_str.split(';'):
        if ':' not in param_spec:
            continue
        
        name, values_str = param_spec.split(':', 1)
        name = name.strip()
        
        # 解析值列表
        values = []
        for v in values_str.split(','):
            v = v.strip()
            # 尝试转换为数字
            try:
                if '.' in v:
                    values.append(float(v))
                else:
                    values.append(int(v))
            except ValueError:
                values.append(v)
        
        parameters[name] = values
    
    return parameters


def main():
    parser = argparse.ArgumentParser(description="Run batch LLM profiling experiments")
    
    # 基础配置
    parser.add_argument("--model", type=str, required=True,
                      help="Model name")
    parser.add_argument("--base-prompt", type=str, default="Hello, how are you?",
                      help="Base prompt for generation")
    parser.add_argument("--profile-mode", type=str, 
                      choices=["inference_only", "nsys", "nsys_ncu"],
                      default="inference_only",
                      help="Profiling mode")
    
    # 参数扫描
    parser.add_argument("--parameter-sweep", type=str,
                      help="Parameter sweep specification (e.g., 'batch_size:1,2,4;max_tokens:50,100')")
    
    # 预设实验
    parser.add_argument("--preset", type=str,
                      choices=["quick", "comprehensive", "scaling"],
                      help="Use preset experiment configurations")
    
    args = parser.parse_args()
    
    # 构建模型路径
    models_dir = "/home/wanghaonan/project/llm_profiling/models"
    model_path = os.path.join(models_dir, args.model)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # 创建基础配置
    base_config = ExperimentConfig(
        model_path=model_path,
        model_name=args.model,
        prompt=args.base_prompt,
        profile_mode=args.profile_mode
    )
    
    # 确定参数范围
    if args.preset:
        parameters = get_preset_parameters(args.preset)
    elif args.parameter_sweep:
        parameters = parse_parameter_ranges(args.parameter_sweep)
    else:
        print("Error: Either --parameter-sweep or --preset must be specified")
        sys.exit(1)
    
    # 打印批量实验信息
    print("=" * 60)
    print("Batch Experiment Configuration")
    print("=" * 60)
    print(f"Model: {base_config.model_name}")
    print(f"Profile Mode: {base_config.profile_mode}")
    print(f"Base Prompt: {base_config.prompt[:50]}...")
    print("\nParameter Sweep:")
    for param, values in parameters.items():
        print(f"  {param}: {values}")
    
    total_experiments = 1
    for values in parameters.values():
        total_experiments *= len(values)
    print(f"\nTotal experiments to run: {total_experiments}")
    print("=" * 60)
    
    # 确认继续
    response = input("\nContinue with batch experiments? (y/n): ")
    if response.lower() != 'y':
        print("Batch experiments cancelled.")
        return
    
    # 运行批量实验
    batch_runner = BatchRunner(base_config)
    results = batch_runner.run_parameter_sweep(parameters)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("Batch Experiment Summary")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['result'].get('status') == 'completed')}")
    print(f"Failed: {sum(1 for r in results if r['result'].get('status') == 'failed')}")
    
    print("\nResults summary saved to: /home/wanghaonan/project/llm_profiling/results/batch_summary.json")
    print("=" * 60)


def get_preset_parameters(preset_name):
    """获取预设参数配置"""
    presets = {
        "quick": {
            "batch_size": [1, 2],
            "max_tokens": [50, 100]
        },
        "comprehensive": {
            "batch_size": [1, 2, 4, 8],
            "max_tokens": [50, 100, 200, 500],
            "temperature": [0.1, 0.5, 0.7, 1.0]
        },
        "scaling": {
            "batch_size": [1, 2, 4, 8, 16],
            "tensor_parallel_size": [1, 2, 4]
        }
    }
    
    return presets.get(preset_name, {})


if __name__ == "__main__":
    main() 