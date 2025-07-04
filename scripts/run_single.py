#!/usr/bin/env python3
"""
单次实验运行脚本
"""
import sys
import os
import argparse
from pathlib import Path

# 添加父目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.config import ExperimentConfig
from profiler.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run single LLM profiling experiment")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True,
                      help="Model name (DeepSeek-R1-Distill-Llama-70B, Qwen2.5-72B-Instruct, etc.)")
    
    # 推理参数
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                      help="Input prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                      help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Batch size for inference")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                      help="Tensor parallel size")
    
    # 分析参数
    parser.add_argument("--profile-mode", type=str, 
                      choices=["inference_only", "nsys", "nsys_ncu"],
                      default="inference_only",
                      help="Profiling mode")
    
    # 其他参数
    parser.add_argument("--experiment-name", type=str, default=None,
                      help="Custom experiment name")
    
    args = parser.parse_args()
    
    # 构建模型路径
    models_dir = "/home/wanghaonan/project/llm_profiling/models"
    model_path = os.path.join(models_dir, args.model)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Available models in {models_dir}:")
        for model in os.listdir(models_dir):
            print(f"  - {model}")
        sys.exit(1)
    
    # 创建配置
    config = ExperimentConfig(
        model_path=model_path,
        model_name=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        profile_mode=args.profile_mode,
        experiment_name=args.experiment_name
    )
    
    # 打印配置信息
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Profile Mode: {config.profile_mode}")
    print(f"Prompt: {config.prompt[:50]}...")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Tensor Parallel Size: {config.tensor_parallel_size}")
    print("=" * 60)
    
    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run()
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Experiment Results")
    print("=" * 60)
    print(f"Status: {results.get('status')}")
    print(f"Output Directory: {runner.output_dir}")
    
    if results.get('status') == 'completed':
        if 'inference_time' in results:
            print(f"Inference Time: {results['inference_time']:.2f} seconds")
        
        print("\nGenerated files:")
        for file in os.listdir(runner.output_dir):
            file_path = os.path.join(runner.output_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {file} ({size:.2f} MB)")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 