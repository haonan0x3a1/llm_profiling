"""
工具函数模块
"""
import os
import json
import subprocess
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def run_command(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """运行shell命令并返回结果"""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        text=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """保存性能指标到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def parse_nsys_stats(stats_file: str) -> Dict[str, Any]:
    """解析nsys统计文件"""
    metrics = {}
    
    if not os.path.exists(stats_file):
        return metrics
    
    with open(stats_file, 'r') as f:
        content = f.read()
    
    # 这里可以添加更复杂的解析逻辑
    # 目前返回基本信息
    metrics['file_size'] = os.path.getsize(stats_file)
    metrics['parse_time'] = datetime.now().isoformat()
    
    return metrics


def get_gpu_info() -> Dict[str, Any]:
    """获取GPU信息"""
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"]
        returncode, stdout, stderr = run_command(cmd)
        
        if returncode == 0:
            gpu_info = []
            for line in stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_info.append({
                        'name': parts[0],
                        'memory': parts[1],
                        'compute_capability': parts[2]
                    })
            return {'gpus': gpu_info}
    except Exception as e:
        return {'error': str(e)}
    
    return {}


def format_prompt_for_display(prompt: str, max_length: int = 50) -> str:
    """格式化prompt用于显示"""
    if len(prompt) <= max_length:
        return prompt
    return prompt[:max_length-3] + "..."


def estimate_vram_usage(model_name: str, batch_size: int, tensor_parallel_size: int) -> float:
    """估算VRAM使用量（GB）"""
    # 简单的估算公式，实际使用中可能需要调整
    model_sizes = {
        "70B": 140,  # 70B模型大约需要140GB（FP16）
        "72B": 144,
        "32B": 64,
        "7B": 14,
    }
    
    for key, size in model_sizes.items():
        if key in model_name:
            base_vram = size / tensor_parallel_size
            # 考虑KV cache和激活值
            return base_vram * (1 + 0.2 * batch_size)
    
    return 0.0  # 未知模型 