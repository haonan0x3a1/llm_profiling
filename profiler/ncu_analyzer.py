"""
Nsight Compute 分析器模块
"""
import os
import json
import re
from typing import Dict, Any, List, Optional
from .utils import run_command, save_metrics, setup_logger


class NcuAnalyzer:
    """Nsight Compute 分析器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = setup_logger("ncu_analyzer", os.path.join(output_dir, "ncu.log"))
    
    def analyze_from_nsys(self, nsys_report_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """基于nsys报告进行ncu分析"""
        self.logger.info("Starting Nsight Compute analysis from nsys report...")
        
        # 从nsys报告中提取kernel信息
        kernels = self._extract_kernels_from_nsys(nsys_report_path)
        
        if not kernels:
            self.logger.warning("No kernels found in nsys report")
            return {"status": "no_kernels_found"}
        
        # 选择要分析的kernel
        target_kernels = self._select_target_kernels(kernels, config.get('ncu_options', {}))
        
        results = {
            "status": "success",
            "analyzed_kernels": [],
            "reports": []
        }
        
        # 分析每个目标kernel
        for i, kernel in enumerate(target_kernels):
            kernel_result = self._profile_kernel(kernel, i, config.get('ncu_options', {}))
            results["analyzed_kernels"].append(kernel_result)
            if kernel_result.get("report_path"):
                results["reports"].append(kernel_result["report_path"])
        
        # 生成综合报告
        self._generate_summary_report(results)
        
        return results
    
    def profile_command(self, cmd: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """直接使用ncu分析命令"""
        self.logger.info("Starting direct Nsight Compute profiling...")
        
        ncu_output = os.path.join(self.output_dir, "ncu_profile")
        ncu_cmd = self._build_ncu_command(ncu_output, config.get('ncu_options', {}))
        
        full_cmd = ncu_cmd + cmd
        
        self.logger.info(f"Running command: {' '.join(full_cmd)}")
        
        returncode, stdout, stderr = run_command(full_cmd, env=os.environ.copy())
        
        # 保存输出
        with open(os.path.join(self.output_dir, "ncu_stdout.txt"), 'w') as f:
            f.write(stdout)
        with open(os.path.join(self.output_dir, "ncu_stderr.txt"), 'w') as f:
            f.write(stderr)
        
        if returncode != 0:
            self.logger.error(f"NCU profiling failed with code {returncode}")
            return {"status": "failed", "error": stderr}
        
        # 解析输出
        metrics = self._parse_ncu_output(stdout)
        save_metrics(metrics, os.path.join(self.output_dir, "ncu_metrics.json"))
        
        return {
            "status": "success",
            "report_path": f"{ncu_output}.ncu-rep",
            "metrics": metrics
        }
    
    def _extract_kernels_from_nsys(self, nsys_report_path: str) -> List[Dict[str, Any]]:
        """从nsys报告中提取kernel信息"""
        kernels = []
        
        try:
            # 使用nsys stats导出kernel列表
            cmd = ["nsys", "stats", nsys_report_path, "-r", "cuda_gpu_kern_sum", "--format", "csv"]
            returncode, stdout, stderr = run_command(cmd)
            
            if returncode != 0:
                self.logger.error(f"Failed to extract kernels: {stderr}")
                return kernels
            
            # 解析CSV输出
            lines = stdout.strip().split('\n')
            if len(lines) < 2:
                return kernels
            
            # 跳过header
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    kernels.append({
                        "name": parts[4].strip('"'),
                        "time_percentage": float(parts[0]) if parts[0] else 0,
                        "instance_count": int(parts[3]) if parts[3] else 0
                    })
            
        except Exception as e:
            self.logger.error(f"Error extracting kernels: {e}")
        
        return kernels
    
    def _select_target_kernels(self, kernels: List[Dict[str, Any]], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """选择要分析的目标kernel"""
        # 按时间百分比排序
        sorted_kernels = sorted(kernels, key=lambda x: x["time_percentage"], reverse=True)
        
        # 选择策略
        kernel_name = options.get('kernel_name')
        if kernel_name:
            # 分析特定kernel
            for kernel in sorted_kernels:
                if kernel_name in kernel["name"]:
                    return [kernel]
            self.logger.warning(f"Kernel {kernel_name} not found")
            return []
        
        # 分析前N个最耗时的kernel
        launch_count = options.get('launch_count', 5)
        return sorted_kernels[:launch_count]
    
    def _build_ncu_command(self, output_prefix: str, options: Dict[str, Any]) -> List[str]:
        """构建ncu命令"""
        cmd = ["ncu"]
        
        # 输出文件
        cmd.extend(["-o", output_prefix])
        
        # 强制覆盖
        cmd.extend(["--force-overwrite", "true"])
        
        # 指标集
        if 'set' in options:
            cmd.extend(["--set", options['set']])
        else:
            cmd.extend(["--set", "full"])
        
        # 特定指标
        metrics = options.get('metrics', [])
        for metric in metrics:
            cmd.extend(["--metrics", metric])
        
        # kernel数量限制
        if 'kernel_count' in options:
            cmd.extend(["-c", str(options['kernel_count'])])
        
        # 特定kernel名称
        if 'kernel_name' in options and options['kernel_name']:
            cmd.extend(["-k", options['kernel_name']])
        
        return cmd
    
    def _profile_kernel(self, kernel: Dict[str, Any], index: int, options: Dict[str, Any]) -> Dict[str, Any]:
        """分析单个kernel"""
        kernel_name = kernel["name"]
        self.logger.info(f"Profiling kernel {index}: {kernel_name}")
        
        # 简化kernel名称用于文件名
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', kernel_name)[:50]
        output_path = os.path.join(self.output_dir, f"kernel_{index}_{safe_name}")
        
        # 构建命令（这里需要实际的执行命令）
        # 注意：这需要能重现kernel执行的命令
        result = {
            "kernel_name": kernel_name,
            "time_percentage": kernel["time_percentage"],
            "status": "skipped",
            "reason": "需要可执行命令来重现kernel"
        }
        
        return result
    
    def _parse_ncu_output(self, output: str) -> Dict[str, Any]:
        """解析ncu输出"""
        metrics = {
            "kernels": [],
            "summary": {}
        }
        
        # 简单的解析逻辑，可根据需要扩展
        lines = output.split('\n')
        current_kernel = None
        
        for line in lines:
            # 检测kernel开始
            if "==PROF==" in line and "Profiling" in line:
                if current_kernel:
                    metrics["kernels"].append(current_kernel)
                current_kernel = {"name": line.split("Profiling")[1].strip()}
            
            # 解析指标
            elif current_kernel and ":" in line:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    current_kernel[key] = value
        
        if current_kernel:
            metrics["kernels"].append(current_kernel)
        
        # 计算总结
        metrics["summary"]["total_kernels"] = len(metrics["kernels"])
        
        return metrics
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """生成综合报告"""
        report = {
            "analysis_time": os.path.getmtime(self.output_dir),
            "total_kernels_analyzed": len(results.get("analyzed_kernels", [])),
            "reports_generated": len(results.get("reports", [])),
            "kernels": results.get("analyzed_kernels", [])
        }
        
        save_metrics(report, os.path.join(self.output_dir, "ncu_summary.json"))
        self.logger.info("Generated summary report")
    
    def export_metrics_comparison(self, output_path: Optional[str] = None):
        """导出kernel性能对比"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "kernel_comparison.csv")
        
        # 读取所有kernel报告
        summary_file = os.path.join(self.output_dir, "ncu_summary.json")
        if not os.path.exists(summary_file):
            self.logger.warning("No summary file found")
            return
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # 导出到CSV
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Kernel Name", "Time %", "Status", "Notes"])
            
            for kernel in summary.get("kernels", []):
                writer.writerow([
                    kernel.get("kernel_name", ""),
                    kernel.get("time_percentage", 0),
                    kernel.get("status", ""),
                    kernel.get("reason", "") if kernel.get("status") == "skipped" else ""
                ])
        
        self.logger.info(f"Exported comparison to: {output_path}") 