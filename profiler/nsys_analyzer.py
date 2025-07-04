"""
Nsight System 分析器模块
"""
import os
import json
import subprocess
from typing import Dict, Any, List, Optional
from .utils import run_command, save_metrics, setup_logger


class NsysAnalyzer:
    """Nsight System 分析器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = setup_logger("nsys_analyzer", os.path.join(output_dir, "nsys.log"))
    
    def profile_inference(self, vllm_cmd: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """使用nsys分析推理过程"""
        self.logger.info("Starting Nsight System profiling...")
        
        # 构建nsys命令
        nsys_output = os.path.join(self.output_dir, "vllm_profile")
        nsys_cmd = self._build_nsys_command(nsys_output, config.get('nsys_options', {}))
        
        # 完整命令
        full_cmd = nsys_cmd + vllm_cmd
        
        self.logger.info(f"Running command: {' '.join(full_cmd)}")
        
        # 执行分析
        returncode, stdout, stderr = run_command(full_cmd, env=os.environ.copy())
        
        # 保存输出
        with open(os.path.join(self.output_dir, "nsys_stdout.txt"), 'w') as f:
            f.write(stdout)
        with open(os.path.join(self.output_dir, "nsys_stderr.txt"), 'w') as f:
            f.write(stderr)
        
        if returncode != 0:
            self.logger.error(f"Nsys profiling failed with code {returncode}")
            self.logger.error(f"Error: {stderr}")
            return {"status": "failed", "error": stderr}
        
        # 分析结果
        results = self._analyze_results(nsys_output)
        results["status"] = "success"
        
        # 生成统计报告
        self._generate_stats_report(nsys_output)
        
        return results
    
    def _build_nsys_command(self, output_prefix: str, options: Dict[str, Any]) -> List[str]:
        """构建nsys命令"""
        cmd = ["nsys", "profile"]
        
        # 输出文件
        cmd.extend(["-o", output_prefix])
        
        # 强制覆盖
        cmd.extend(["--force-overwrite", "true"])
        
        # 选项
        if options.get('cuda', True):
            cmd.extend(["-t", "cuda,osrt,nvtx"])
        
        if options.get('gpu_metrics', True):
            cmd.append("--gpu-metrics-device=all")
        
        if options.get('stats', True):
            cmd.extend(["--stats", "true"])
        
        # 其他选项
        if 'sample_rate' in options:
            cmd.extend(["--sampling-frequency", str(options['sample_rate'])])
        
        return cmd
    
    def _analyze_results(self, output_prefix: str) -> Dict[str, Any]:
        """分析nsys结果"""
        results: Dict[str, Any] = {
            "nsys_report": f"{output_prefix}.nsys-rep",
            "sqlite_db": f"{output_prefix}.sqlite"
        }
        
        # 检查文件是否生成
        for key, filepath in list(results.items()):
            if os.path.exists(filepath):
                results[f"{key}_size"] = os.path.getsize(filepath)
                self.logger.info(f"Generated {key}: {filepath} ({results[f'{key}_size']} bytes)")
            else:
                self.logger.warning(f"File not found: {filepath}")
        
        return results
    
    def _generate_stats_report(self, output_prefix: str):
        """生成统计报告"""
        try:
            # 导出统计信息
            stats_file = os.path.join(self.output_dir, "nsys_stats.txt")
            cmd = ["nsys", "stats", f"{output_prefix}.nsys-rep", "-r", "cuda_api_sum,cuda_gpu_kern_sum,osrt_sum"]
            
            returncode, stdout, stderr = run_command(cmd)
            
            if returncode == 0:
                with open(stats_file, 'w') as f:
                    f.write(stdout)
                self.logger.info(f"Stats report saved to: {stats_file}")
                
                # 解析并保存关键指标
                metrics = self._parse_stats(stdout)
                save_metrics(metrics, os.path.join(self.output_dir, "nsys_metrics.json"))
            else:
                self.logger.error(f"Failed to generate stats: {stderr}")
                
        except Exception as e:
            self.logger.error(f"Error generating stats report: {e}")
    
    def _parse_stats(self, stats_output: str) -> Dict[str, Any]:
        """解析nsys统计输出"""
        metrics = {
            "cuda_kernels": [],
            "cuda_api_calls": [],
            "summary": {}
        }
        
        lines = stats_output.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # 检测section
            if "CUDA API Statistics" in line:
                current_section = "cuda_api"
            elif "CUDA Kernel Statistics" in line:
                current_section = "cuda_kernel"
            elif "Operating System Runtime API Statistics" in line:
                current_section = "osrt"
            
            # 解析数据行（简化版本，可根据需要扩展）
            if current_section and line and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 5 and parts[0].replace('.', '').isdigit():
                    if current_section == "cuda_kernel":
                        metrics["cuda_kernels"].append({
                            "name": ' '.join(parts[4:]),
                            "time_percentage": float(parts[0]),
                            "total_time_ns": int(parts[1].replace(',', ''))
                        })
                    elif current_section == "cuda_api":
                        metrics["cuda_api_calls"].append({
                            "name": ' '.join(parts[4:]),
                            "time_percentage": float(parts[0]),
                            "total_time_ns": int(parts[1].replace(',', ''))
                        })
        
        # 计算总结信息
        if metrics["cuda_kernels"]:
            metrics["summary"]["total_kernel_time_ns"] = sum(k["total_time_ns"] for k in metrics["cuda_kernels"])
            metrics["summary"]["top_kernels"] = sorted(metrics["cuda_kernels"], 
                                                      key=lambda x: x["time_percentage"], 
                                                      reverse=True)[:5]
        
        return metrics
    
    def export_to_csv(self, csv_path: Optional[str] = None):
        """导出分析结果到CSV格式"""
        if csv_path is None:
            csv_path = os.path.join(self.output_dir, "nsys_summary.csv")
        
        # 读取metrics
        metrics_file = os.path.join(self.output_dir, "nsys_metrics.json")
        if not os.path.exists(metrics_file):
            self.logger.warning("No metrics file found")
            return
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # 导出kernel信息到CSV
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Kernel Name", "Time %", "Total Time (ns)"])
            
            for kernel in metrics.get("cuda_kernels", []):
                writer.writerow([
                    kernel["name"],
                    kernel["time_percentage"],
                    kernel["total_time_ns"]
                ])
        
        self.logger.info(f"Exported results to: {csv_path}") 