"""
主运行器模块
"""
import os
import sys
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import ExperimentConfig, ConfigManager
from .nsys_analyzer import NsysAnalyzer
from .ncu_analyzer import NcuAnalyzer
from .visualizer import Visualizer
from .utils import setup_logger, get_gpu_info, save_metrics, format_prompt_for_display, estimate_vram_usage


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = ConfigManager.create_experiment_dir(config)
        self.logger = setup_logger("experiment_runner", 
                                 os.path.join(self.output_dir, "experiment.log"))
        
        # 保存配置
        ConfigManager.save_config(config, os.path.join(self.output_dir, "config.yaml"))
    
    def run(self) -> Dict[str, Any]:
        """运行实验"""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Profile mode: {self.config.profile_mode}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # 记录GPU信息
        gpu_info = get_gpu_info()
        save_metrics(gpu_info, os.path.join(self.output_dir, "gpu_info.json"))
        
        # 检查VRAM需求
        estimated_vram = estimate_vram_usage(
            self.config.model_name,
            self.config.batch_size,
            self.config.tensor_parallel_size
        )
        self.logger.info(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
        
        # 构建vLLM命令
        vllm_cmd = self._build_vllm_command()
        
        # 根据模式运行分析
        results = {"start_time": datetime.now().isoformat()}
        
        try:
            if self.config.profile_mode == "inference_only":
                results.update(self._run_inference_only(vllm_cmd))
            elif self.config.profile_mode == "nsys":
                results.update(self._run_nsys_profiling(vllm_cmd))
            elif self.config.profile_mode == "nsys_ncu":
                results.update(self._run_full_profiling(vllm_cmd))
            else:
                raise ValueError(f"Unknown profile mode: {self.config.profile_mode}")
            
            results["status"] = "completed"
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        results["end_time"] = datetime.now().isoformat()
        
        # 保存结果摘要
        save_metrics(results, os.path.join(self.output_dir, "summary.json"))
        
        # 生成可视化
        if results.get("status") == "completed" and self.config.profile_mode != "inference_only":
            self._generate_visualizations()
        
        self.logger.info(f"Experiment completed: {results.get('status')}")
        return results
    
    def _build_vllm_command(self) -> List[str]:
        """构建vLLM推理命令"""
        # 创建Python脚本来运行vLLM
        # 设置可见的GPU数量，确保与tensor_parallel_size匹配
        gpu_ids = ",".join(str(i) for i in range(self.config.tensor_parallel_size))
        script_content = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_ids}"  # 设置可见GPU数量

from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="{self.config.model_path}",
    tensor_parallel_size={self.config.tensor_parallel_size},
    trust_remote_code=True
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature={self.config.temperature},
    max_tokens={self.config.max_tokens}
)

# 运行推理
prompts = ["{self.config.prompt}"] * {self.config.batch_size}
outputs = llm.generate(prompts, sampling_params)

# 保存输出
with open("{os.path.join(self.output_dir, 'generated_output.txt')}", "w") as f:
    for i, output in enumerate(outputs):
        f.write(f"Prompt {{i+1}}: {{output.prompt}}\\n")
        f.write(f"Generated: {{output.outputs[0].text}}\\n")
        f.write("-" * 80 + "\\n")

print("Inference completed successfully")
'''
        
        # 保存脚本
        script_path = os.path.join(self.output_dir, "inference_script.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 返回命令
        return [sys.executable, script_path]
    
    def _run_inference_only(self, vllm_cmd: List[str]) -> Dict[str, Any]:
        """仅运行推理，不进行性能分析"""
        self.logger.info("Running inference only (no profiling)")
        
        from .utils import run_command
        
        start_time = time.time()
        returncode, stdout, stderr = run_command(vllm_cmd, cwd=self.output_dir)
        end_time = time.time()
        
        # 保存输出
        with open(os.path.join(self.output_dir, "stdout.txt"), 'w') as f:
            f.write(stdout)
        with open(os.path.join(self.output_dir, "stderr.txt"), 'w') as f:
            f.write(stderr)
        
        results = {
            "inference_time": end_time - start_time,
            "returncode": returncode
        }
        
        if returncode == 0:
            self.logger.info(f"Inference completed in {results['inference_time']:.2f} seconds")
        else:
            self.logger.error(f"Inference failed with code {returncode}")
            results["error"] = stderr
        
        # 保存基本指标
        metrics = {
            "model": self.config.model_name,
            "prompt": format_prompt_for_display(self.config.prompt),
            "batch_size": self.config.batch_size,
            "max_tokens": self.config.max_tokens,
            "inference_time": results["inference_time"],
            "tokens_per_second": (self.config.max_tokens * self.config.batch_size) / results["inference_time"] if results["inference_time"] > 0 else 0
        }
        save_metrics(metrics, os.path.join(self.output_dir, "metrics.json"))
        
        return results
    
    def _run_nsys_profiling(self, vllm_cmd: List[str]) -> Dict[str, Any]:
        """运行Nsight System分析"""
        self.logger.info("Running Nsight System profiling")
        
        analyzer = NsysAnalyzer(self.output_dir)
        results = analyzer.profile_inference(vllm_cmd, {
            "nsys_options": self.config.nsys_options
        })
        
        # 导出CSV
        if results.get("status") == "success":
            analyzer.export_to_csv()
        
        return results
    
    def _run_full_profiling(self, vllm_cmd: List[str]) -> Dict[str, Any]:
        """运行完整分析（Nsight System + Nsight Compute）"""
        self.logger.info("Running full profiling (nsys + ncu)")
        
        # 首先运行nsys分析
        nsys_results = self._run_nsys_profiling(vllm_cmd)
        
        if nsys_results.get("status") != "success":
            return nsys_results
        
        # 然后基于nsys结果运行ncu分析
        nsys_report = nsys_results.get("nsys_report")
        if nsys_report and os.path.exists(nsys_report):
            ncu_analyzer = NcuAnalyzer(self.output_dir)
            ncu_results = ncu_analyzer.analyze_from_nsys(nsys_report, {
                "ncu_options": self.config.ncu_options
            })
            
            # 合并结果
            nsys_results["ncu_analysis"] = ncu_results
            
            # 导出对比
            if ncu_results.get("status") == "success":
                ncu_analyzer.export_metrics_comparison()
        
        return nsys_results
    
    def _generate_visualizations(self):
        """生成可视化结果"""
        self.logger.info("Generating visualizations")
        
        try:
            visualizer = Visualizer(self.output_dir)
            visualizer.create_all_visualizations()
            
            # 生成HTML报告
            html_report = visualizer.generate_report_html()
            self.logger.info(f"Generated HTML report: {html_report}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")


class BatchRunner:
    """批量实验运行器"""
    
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.results = []
    
    def run_parameter_sweep(self, parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """运行参数扫描"""
        from itertools import product
        
        # 生成所有参数组合
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]
        
        for values in product(*param_values):
            # 创建新配置
            config_dict = self.base_config.__dict__.copy()
            
            # 更新参数
            for name, value in zip(param_names, values):
                config_dict[name] = value
            
            # 创建新的配置对象
            config = ExperimentConfig(**config_dict)
            
            # 更新实验名称
            param_str = "_".join([f"{name}{value}" for name, value in zip(param_names, values)])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.experiment_name = f"{config.model_name}_{param_str}_{timestamp}"
            config.output_dir = None  # 让它自动生成
            
            # 运行实验
            runner = ExperimentRunner(config)
            result = runner.run()
            
            self.results.append({
                "config": config_dict,
                "result": result
            })
        
        # 保存批量结果摘要
        self._save_batch_summary()
        
        return self.results
    
    def _save_batch_summary(self):
        """保存批量实验摘要"""
        summary = {
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r["result"].get("status") == "completed"),
            "failed": sum(1 for r in self.results if r["result"].get("status") == "failed"),
            "experiments": []
        }
        
        for exp in self.results:
            summary["experiments"].append({
                "name": exp["config"].get("experiment_name", "unknown"),
                "status": exp["result"].get("status"),
                "output_dir": exp["config"].get("output_dir"),
                "key_params": {
                    k: v for k, v in exp["config"].items() 
                    if k in ["prompt", "batch_size", "max_tokens", "tensor_parallel_size"]
                }
            })
        
        # 保存到results目录
        summary_path = "/home/wanghaonan/project/llm_profiling/results/batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Batch summary saved to: {summary_path}") 