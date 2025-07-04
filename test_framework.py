#!/usr/bin/env python3
"""
框架功能测试脚本
"""
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试所有模块是否可以正常导入"""
    print("🧪 测试模块导入...")
    
    try:
        from profiler.config import ExperimentConfig, ConfigManager
        print("✅ config模块导入成功")
        
        from profiler.utils import setup_logger, get_gpu_info
        print("✅ utils模块导入成功")
        
        from profiler.nsys_analyzer import NsysAnalyzer
        print("✅ nsys_analyzer模块导入成功")
        
        from profiler.ncu_analyzer import NcuAnalyzer
        print("✅ ncu_analyzer模块导入成功")
        
        from profiler.runner import ExperimentRunner, BatchRunner
        print("✅ runner模块导入成功")
        
        # 可视化模块可能因为matplotlib缺失而失败，这是正常的
        try:
            from profiler.visualizer import Visualizer
            print("✅ visualizer模块导入成功")
        except ImportError as e:
            print(f"⚠️  visualizer模块导入失败（可能缺少matplotlib）: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_config():
    """测试配置系统"""
    print("\n🧪 测试配置系统...")
    
    try:
        from profiler.config import ExperimentConfig, ConfigManager
        
        # 测试ExperimentConfig
        config = ExperimentConfig(
            model_path="/test/path",
            model_name="test_model"
        )
        print(f"✅ 创建配置成功: {config.experiment_name}")
        
        # 测试默认配置加载
        if os.path.exists("configs/default.yaml"):
            default_config = ConfigManager.load_config("configs/default.yaml")
            print(f"✅ 加载默认配置成功: {default_config.model_name}")
        else:
            print("⚠️  默认配置文件不存在")
        
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def test_gpu_info():
    """测试GPU信息获取"""
    print("\n🧪 测试GPU信息获取...")
    
    try:
        from profiler.utils import get_gpu_info
        gpu_info = get_gpu_info()
        
        if 'gpus' in gpu_info:
            print(f"✅ 检测到 {len(gpu_info['gpus'])} 个GPU:")
            for i, gpu in enumerate(gpu_info['gpus']):
                print(f"   GPU {i}: {gpu.get('name', 'Unknown')}")
        elif 'error' in gpu_info:
            print(f"⚠️  GPU信息获取警告: {gpu_info['error']}")
        else:
            print("⚠️  未检测到GPU信息")
        
        return True
    except Exception as e:
        print(f"❌ GPU信息测试失败: {e}")
        return False


def test_model_paths():
    """测试模型路径"""
    print("\n🧪 测试模型路径...")
    
    models_dir = "/home/wanghaonan/project/llm_profiling/models"
    if os.path.exists(models_dir):
        models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        print(f"✅ 模型目录存在，找到 {len(models)} 个模型:")
        for model in models:
            model_path = os.path.join(models_dir, model)
            size_info = ""
            try:
                # 简单检查模型目录大小
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
                size_gb = total_size / (1024**3)
                size_info = f" ({size_gb:.1f} GB)"
            except:
                pass
            print(f"   - {model}{size_info}")
    else:
        print(f"⚠️  模型目录不存在: {models_dir}")
    
    return True


def test_results_directory():
    """测试结果目录"""
    print("\n🧪 测试结果目录...")
    
    results_dir = "/home/wanghaonan/project/llm_profiling/results"
    viz_dir = "/home/wanghaonan/project/llm_profiling/results/visualization"
    
    try:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        print(f"✅ 结果目录创建成功: {results_dir}")
        print(f"✅ 可视化目录创建成功: {viz_dir}")
        return True
    except Exception as e:
        print(f"❌ 目录创建失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("LLM Profiling Framework - 功能测试")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_gpu_info,
        test_model_paths,
        test_results_directory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！框架准备就绪。")
        print("\n📖 使用指南:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 安装vLLM: pip install vllm")
        print("3. 运行简单测试: python scripts/run_single.py --model Qwen2.5-72B-Instruct")
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 