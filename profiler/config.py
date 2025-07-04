"""
配置管理模块
"""
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperimentConfig:
    """实验配置类"""
    # 模型配置
    model_path: str
    model_name: str = ""
    
    # 推理配置
    prompt: str = "Hello, how are you?"
    max_tokens: int = 100
    temperature: float = 0.7
    tensor_parallel_size: int = 1
    batch_size: int = 1
    
    # 分析配置
    profile_mode: str = "inference_only"  # inference_only, nsys, nsys_ncu
    nsys_options: Dict[str, Any] = field(default_factory=lambda: {
        "cuda": True,
        "gpu_metrics": True,
        "stats": True
    })
    ncu_options: Dict[str, Any] = field(default_factory=lambda: {
        "kernel_name": None,  # 分析特定kernel，None表示全部
        "launch_count": 10,   # 分析前N个kernel launches
        "metrics": ["sm__cycles_elapsed.avg", "sm__cycles_elapsed.avg.per_second"]
    })
    
    # 输出配置
    output_dir: Optional[str] = None
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.model_name:
            self.model_name = os.path.basename(self.model_path)
        
        if not self.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.model_name}_{self.profile_mode}_{timestamp}"
        
        if not self.output_dir:
            base_dir = "/home/wanghaonan/project/llm_profiling/results"
            self.output_dir = os.path.join(base_dir, self.experiment_name)


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(**config_dict)
    
    @staticmethod
    def save_config(config: ExperimentConfig, save_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            "model_path": config.model_path,
            "model_name": config.model_name,
            "prompt": config.prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "tensor_parallel_size": config.tensor_parallel_size,
            "batch_size": config.batch_size,
            "profile_mode": config.profile_mode,
            "nsys_options": config.nsys_options,
            "ncu_options": config.ncu_options,
            "output_dir": config.output_dir,
            "experiment_name": config.experiment_name
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def create_experiment_dir(config: ExperimentConfig) -> str:
        """创建实验目录"""
        if config.output_dir is None:
            raise ValueError("Output directory not set")
        os.makedirs(config.output_dir, exist_ok=True)
        return config.output_dir 