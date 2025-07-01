import torch
from vllm import LLM, SamplingParams
import time
import os
import argparse
import sys
from datetime import datetime
import shutil
import subprocess
import signal
import gc

# --- 配置常量 ---
MODELS_BASE_DIR = "/home/wanghaonan/project/llm_profiling/models/"
RESULTS_BASE_DIR = "/home/wanghaonan/project/llm_profiling/results/"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_PROMPT = "The {model_name} model is a large language model. Write a short paragraph about its potential applications."
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
NUM_REQUESTS_TO_PROFILE = 1

# --- NVTX 辅助类 ---
class NvtxRange:
    """NVTX 范围的上下文管理器"""
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.nvtx_available = False

        if self.enabled and torch.cuda.is_available():
            try:
                getattr(torch.cuda.nvtx, 'range_push')
                self.nvtx_available = True
            except AttributeError:
                pass

    def __enter__(self):
        if self.enabled and self.nvtx_available:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.nvtx_available:
            torch.cuda.nvtx.range_pop()

# --- 工具函数 ---
def list_available_models(base_dir: str) -> list:
    """列出可用模型文件夹"""
    return sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

def select_model_and_gpus():
    """交互式选择模型和GPU数量"""
    # 选择模型
    available_models = list_available_models(MODELS_BASE_DIR)
    if not available_models:
        print(f"错误: {MODELS_BASE_DIR} 中没有找到模型")
        return None, None, None

    print("\n--- 可用模型列表 ---")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model}")

    while True:
        try:
            choice = int(input(f"请选择模型 (1-{len(available_models)}): "))
            if 1 <= choice <= len(available_models):
                selected_model = available_models[choice-1]
                model_path = os.path.join(MODELS_BASE_DIR, selected_model)
                break
            print("无效选择")
        except ValueError:
            print("请输入数字")

    # 选择GPU数量
    while True:
        try:
            num_gpus = int(input("GPU数量 (例如 1, 2, 4): "))
            if num_gpus > 0:
                print(f"使用 {num_gpus} 个GPU")
                print(f"提示: 请通过 CUDA_VISIBLE_DEVICES 环境变量指定GPU")
                return selected_model, model_path, num_gpus
            print("GPU数量必须 > 0")
        except ValueError:
            print("请输入整数")

    return None, None, None

def safe_shutdown(llm_instance):
    """安全关闭LLM实例并释放资源"""
    if llm_instance:
        try:
            # 显式关闭引擎释放资源
            if hasattr(llm_instance, 'llm_engine') and llm_instance.llm_engine:
                llm_instance.llm_engine.shutdown()
            del llm_instance
        except Exception as e:
            print(f"关闭引擎时出错: {e}")
        finally:
            # 清空CUDA缓存
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()
            print("资源已安全释放")

def create_result_dir(model_name, tp_size, with_nsys=False):
    """创建结果目录并返回路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{model_name}_TP{tp_size}"
    if with_nsys:
        dir_name += "_nsys"
    
    result_dir = os.path.join(RESULTS_BASE_DIR, dir_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def save_config_info(result_dir, config):
    """保存配置信息到文件"""
    config_file = os.path.join(result_dir, "config.txt")
    with open(config_file, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    return config_file

def run_with_nsys(selected_model, model_path, tensor_parallel_size):
    """使用nsys运行分析"""
    # 创建唯一结果目录
    result_dir = create_result_dir(selected_model, tensor_parallel_size, with_nsys=True)
    
    # 构造nsys命令 (使用标准选项避免歧义)
    nsys_output = os.path.join(result_dir, "vllm_profile")
    python_script = os.path.abspath(__file__)
    
    # 简化命令避免选项歧义
    nsys_command = [
        "nsys", "profile", 
        "-t", "cuda,nvtx",  # 只保留基本CUDA和NVTX跟踪
        "--stats", "true",  # 收集统计数据
        "-o", nsys_output,
        "--force-overwrite", "true",
        "python", python_script, 
        "--model", selected_model, 
        "--gpus", str(tensor_parallel_size),
        "--result-dir", result_dir
    ]
    
    print(f"使用Nsys分析: {' '.join(nsys_command)}")
    
    # 准备环境变量（特别是CUDA_VISIBLE_DEVICES）
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" in env:
        print(f"使用CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    
    try:
        with open(os.path.join(result_dir, "nsys.log"), "w") as log_file:
            process = subprocess.Popen(
                nsys_command,
                stdout=log_file,
                stderr=log_file,
                env=env
            )
            # 等待进程完成
            return_code = process.wait()
            if return_code == 0:
                print(f"Nsys分析成功完成! 报告保存于: {nsys_output}.nsys-rep")
                return True
            else:
                print(f"Nsys分析失败，返回码: {return_code}")
                return False
    except Exception as e:
        print(f"运行Nsys时出错: {str(e)}")
        return False

# --- 主分析函数 ---
def run_llm_inference(selected_model_name, model_path, tensor_parallel_size, result_dir=None):
    """执行性能分析"""
    # 创建结果目录
    if not result_dir:
        result_dir = create_result_dir(selected_model_name, tensor_parallel_size)
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存配置信息
    config_info = {
        "timestamp": datetime.now().isoformat(),
        "model": selected_model_name,
        "model_path": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "num_requests": NUM_REQUESTS_TO_PROFILE,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    save_config_info(result_dir, config_info)
    
    print("="*80)
    print(f"开始分析: {selected_model_name} (TP={tensor_parallel_size})")
    print(f"结果目录: {result_dir}")
    print("="*80)
    
    # 环境信息
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"检测到 {torch.cuda.device_count()} 个GPU:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    
    # 1. 加载模型
    llm = None
    nvtx_load_range = f"vLLM_Load_{selected_model_name}_TP{tensor_parallel_size}"
    with NvtxRange(nvtx_load_range, enabled=True):
        print(f"\n[1] 加载模型: {model_path}")
        print(f"    Tensor Parallel: {tensor_parallel_size}")
        print(f"    GPU显存利用率: {DEFAULT_GPU_MEMORY_UTILIZATION}")
        
        try:
            start_time = time.time()
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
                dtype="auto",
                trust_remote_code=True,
                enforce_eager=False
            )
            load_time = time.time() - start_time
            print(f"模型加载成功, 耗时: {load_time:.2f}秒")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("可能原因: 显存不足、模型路径错误或依赖缺失")
            return False
    
    # 2. 准备输入
    prompt_text = DEFAULT_PROMPT.format(model_name=selected_model_name)
    sampling_params = SamplingParams(
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_tokens=DEFAULT_MAX_NEW_TOKENS
    )
    prompts = [prompt_text]
    
    print(f"\n[2] 输入准备")
    print(f"    提示: {prompt_text[:100]}...")
    
    # 3. 预热运行
    nvtx_warmup_range = f"vLLM_Warmup_{selected_model_name}_TP{tensor_parallel_size}"
    with NvtxRange(nvtx_warmup_range, enabled=True):
        print("\n[3] 执行预热推理...")
        try:
            _ = llm.generate(prompts, sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("预热完成")
        except Exception as e:
            print(f"预热运行错误: {e}")
            return False
    
    # 4. 分析运行
    print(f"\n[4] 执行 {NUM_REQUESTS_TO_PROFILE} 次推理分析...")
    total_time = 0
    total_tokens = 0
    output_text = ""
    
    for i in range(NUM_REQUESTS_TO_PROFILE):
        nvtx_range = f"vLLM_ProfiledReq_{i+1}_{selected_model_name}_TP{tensor_parallel_size}"
        with NvtxRange(nvtx_range, enabled=True):
            print(f"    执行请求 {i+1}/{NUM_REQUESTS_TO_PROFILE}...")
            start_time = time.time()
            
            try:
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                duration = time.time() - start_time
                total_time += duration
                
                if outputs:
                    tokens = len(outputs[0].outputs[0].token_ids)
                    total_tokens += tokens
                    print(f"    完成, 耗时: {duration:.3f}秒, 生成 {tokens} tokens")
                    
                    # 保存第一个请求的输出
                    if i == 0:
                        output_text = outputs[0].outputs[0].text
                else:
                    print(f"    请求 {i+1} 没有输出")
            except Exception as e:
                print(f"    请求 {i+1} 失败: {e}")
                continue
    
    # 保存输出
    output_file = os.path.join(result_dir, "generated_output.txt")
    with open(output_file, "w") as f:
        f.write(output_text)
        print(f"    生成文本已保存: {output_file}")
    
    # 5. 性能总结
    print("\n[5] 性能总结:")
    print(f"    模型: {selected_model_name}, TP: {tensor_parallel_size}")
    if total_tokens > 0:
        avg_time = total_time / NUM_REQUESTS_TO_PROFILE
        tokens_per_sec = total_tokens / total_time
        ms_per_token = 1000 * total_time / total_tokens
        
        print(f"    平均请求时间: {avg_time:.3f}秒")
        print(f"    总生成token数: {total_tokens}")
        print(f"    吞吐量: {tokens_per_sec:.2f} tokens/秒")
        print(f"    延迟: {ms_per_token:.2f} ms/token")
        
        # 保存性能指标
        metrics_file = os.path.join(result_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"total_requests: {NUM_REQUESTS_TO_PROFILE}\n")
            f.write(f"total_time: {total_time:.3f}\n")
            f.write(f"total_tokens: {total_tokens}\n")
            f.write(f"avg_time_per_request: {avg_time:.3f}\n")
            f.write(f"throughput: {tokens_per_sec:.2f}\n")
            f.write(f"latency_per_token: {ms_per_token:.2f}\n")
    else:
        print("    没有生成token，无法计算性能指标")
    
    # 6. 安全关闭
    print("\n[6] 清理资源...")
    safe_shutdown(llm)
    print("分析完成")
    
    return True

def main():
    """主函数处理命令行参数"""
    parser = argparse.ArgumentParser(description="LLM 推理性能分析工具")
    parser.add_argument("--nsys", action="store_true", help="使用 Nsight Systems 进行性能分析")
    parser.add_argument("--model", type=str, help="直接指定模型名称")
    parser.add_argument("--gpus", type=int, help="直接指定GPU数量")
    parser.add_argument("--result-dir", type=str, help="指定结果目录路径")
    args = parser.parse_args()
    
    # 环境检查
    if not torch.cuda.is_available():
        print("错误: 没有可用的CUDA设备")
        return
    
    # 选择模型和GPU
    if args.model and args.gpus:
        selected_model = args.model
        model_path = os.path.join(MODELS_BASE_DIR, selected_model)
        gpu_count = args.gpus
    else:
        selected_model, model_path, gpu_count = select_model_and_gpus()
        if not model_path:
            return
    
    # 处理Nsys分析
    if args.nsys:
        success = run_with_nsys(selected_model, model_path, gpu_count)
        if success:
            print(f"Nsys分析成功完成!")
        else:
            print(f"Nsys分析失败")
        return
    
    # 执行性能分析
    success = run_llm_inference(
        selected_model_name=selected_model,
        model_path=model_path,
        tensor_parallel_size=gpu_count,
        result_dir=args.result_dir
    )
    
    if success:
        print(f"分析成功完成!")
    else:
        print("分析过程中出现错误")

if __name__ == "__main__":
    main()