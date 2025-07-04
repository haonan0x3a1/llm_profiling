"""
可视化模块
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, List, Optional
import numpy as np


class Visualizer:
    """性能数据可视化器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.viz_dir = "/home/wanghaonan/project/llm_profiling/results/visualization"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        # 读取metrics
        nsys_metrics = self._load_metrics("nsys_metrics.json")
        ncu_metrics = self._load_metrics("ncu_metrics.json")
        
        if nsys_metrics:
            self.plot_kernel_time_distribution(nsys_metrics)
            self.plot_cuda_api_usage(nsys_metrics)
        
        if ncu_metrics:
            self.plot_kernel_comparison(ncu_metrics)
        
        # 创建综合报告
        self.create_summary_dashboard(nsys_metrics, ncu_metrics)
    
    def _load_metrics(self, filename: str) -> Optional[Dict[str, Any]]:
        """加载指标文件"""
        filepath = os.path.join(self.results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def plot_kernel_time_distribution(self, metrics: Dict[str, Any]):
        """绘制kernel时间分布图"""
        kernels = metrics.get("cuda_kernels", [])[:10]  # Top 10
        
        if not kernels:
            return
        
        # 准备数据
        names = [k["name"][:40] + "..." if len(k["name"]) > 40 else k["name"] for k in kernels]
        percentages = [k["time_percentage"] for k in kernels]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        bars = plt.barh(names, percentages)
        
        # 设置颜色渐变
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Time Percentage (%)')
        plt.title('Top 10 CUDA Kernels by Time')
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.viz_dir, f"{os.path.basename(self.results_dir)}_kernel_time.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cuda_api_usage(self, metrics: Dict[str, Any]):
        """绘制CUDA API使用图"""
        api_calls = metrics.get("cuda_api_calls", [])[:10]
        
        if not api_calls:
            return
        
        # 准备数据
        names = [call["name"] for call in api_calls]
        percentages = [call["time_percentage"] for call in api_calls]
        
        # 创建饼图
        plt.figure(figsize=(10, 8))
        
        # 处理小百分比
        threshold = 2.0
        small_slices = []
        large_slices = []
        
        for i, (name, pct) in enumerate(zip(names, percentages)):
            if pct < threshold:
                small_slices.append((name, pct))
            else:
                large_slices.append((name, pct))
        
        if small_slices:
            other_pct = sum(pct for _, pct in small_slices)
            large_slices.append(("Others", other_pct))
        
        if large_slices:
            labels = [name for name, _ in large_slices]
            sizes = [pct for _, pct in large_slices]
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('CUDA API Call Time Distribution')
            
            # 保存
            output_path = os.path.join(self.viz_dir, f"{os.path.basename(self.results_dir)}_cuda_api.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_kernel_comparison(self, metrics: Dict[str, Any]):
        """绘制kernel性能对比图"""
        kernels = metrics.get("kernels", [])
        
        if not kernels:
            return
        
        # 这里可以添加更详细的kernel对比可视化
        # 目前创建一个简单的状态分布图
        status_counts = {}
        for kernel in kernels:
            status = kernel.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            plt.figure(figsize=(8, 6))
            plt.bar(status_counts.keys(), status_counts.values())
            plt.xlabel('Status')
            plt.ylabel('Count')
            plt.title('Kernel Analysis Status Distribution')
            
            output_path = os.path.join(self.viz_dir, f"{os.path.basename(self.results_dir)}_kernel_status.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_dashboard(self, nsys_metrics: Optional[Dict[str, Any]], 
                               ncu_metrics: Optional[Dict[str, Any]]):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(16, 10))
        
        # 创建子图网格
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 总体统计
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_summary_stats(ax1, nsys_metrics, ncu_metrics)
        
        # 2. Top Kernels
        if nsys_metrics and nsys_metrics.get("cuda_kernels"):
            ax2 = fig.add_subplot(gs[1, :2])
            self._plot_top_kernels_mini(ax2, nsys_metrics)
        
        # 3. 性能指标
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_performance_metrics(ax3, nsys_metrics)
        
        # 4. 时间线（占位）
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_timeline_placeholder(ax4)
        
        plt.suptitle(f'LLM Profiling Summary - {os.path.basename(self.results_dir)}', fontsize=16)
        
        # 保存
        output_path = os.path.join(self.viz_dir, f"{os.path.basename(self.results_dir)}_dashboard.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_stats(self, ax, nsys_metrics, ncu_metrics):
        """绘制总体统计信息"""
        ax.axis('off')
        
        stats_text = "Performance Summary\n" + "="*50 + "\n\n"
        
        if nsys_metrics:
            total_kernels = len(nsys_metrics.get("cuda_kernels", []))
            total_api_calls = len(nsys_metrics.get("cuda_api_calls", []))
            stats_text += f"Total CUDA Kernels: {total_kernels}\n"
            stats_text += f"Total CUDA API Calls: {total_api_calls}\n"
            
            if nsys_metrics.get("summary", {}).get("total_kernel_time_ns"):
                total_time_ms = nsys_metrics["summary"]["total_kernel_time_ns"] / 1e6
                stats_text += f"Total Kernel Time: {total_time_ms:.2f} ms\n"
        
        if ncu_metrics:
            analyzed_kernels = len(ncu_metrics.get("kernels", []))
            stats_text += f"\nKernels Analyzed by NCU: {analyzed_kernels}\n"
        
        ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center')
    
    def _plot_top_kernels_mini(self, ax, metrics):
        """绘制迷你版top kernels图"""
        kernels = metrics.get("cuda_kernels", [])[:5]
        
        if kernels:
            names = [f"Kernel {i+1}" for i in range(len(kernels))]
            percentages = [k["time_percentage"] for k in kernels]
            
            bars = ax.bar(names, percentages)
            ax.set_ylabel('Time %')
            ax.set_title('Top 5 Kernels by Time')
            
            # 添加实际kernel名称作为注释
            for i, kernel in enumerate(kernels):
                name = kernel["name"][:30] + "..." if len(kernel["name"]) > 30 else kernel["name"]
                ax.text(i, percentages[i] + 0.5, name, rotation=45, ha='left', fontsize=8)
    
    def _plot_performance_metrics(self, ax, metrics):
        """绘制性能指标"""
        ax.axis('off')
        
        if not metrics:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            return
        
        # 创建一个简单的指标表
        metrics_text = "Key Metrics\n" + "-"*20 + "\n"
        
        if metrics.get("summary", {}).get("top_kernels"):
            top_kernel = metrics["summary"]["top_kernels"][0]
            metrics_text += f"Top Kernel:\n{top_kernel['name'][:25]}...\n"
            metrics_text += f"Time: {top_kernel['time_percentage']:.1f}%\n"
        
        ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center')
    
    def _plot_timeline_placeholder(self, ax):
        """绘制时间线占位图"""
        ax.text(0.5, 0.5, 'Timeline visualization placeholder\n(Can be implemented with detailed trace data)',
                ha='center', va='center', fontsize=12, style='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time')
        ax.set_title('Execution Timeline')
    
    def generate_report_html(self):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Profiling Report - {os.path.basename(self.results_dir)}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 5px 0; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>LLM Profiling Report</h1>
            <div class="section">
                <h2>Experiment: {os.path.basename(self.results_dir)}</h2>
                <p>Generated at: {os.path.getctime(self.results_dir)}</p>
            </div>
        """
        
        # 添加图片
        viz_files = os.listdir(self.viz_dir)
        experiment_files = [f for f in viz_files if os.path.basename(self.results_dir) in f]
        
        if experiment_files:
            html_content += '<div class="section"><h2>Visualizations</h2>'
            for img_file in experiment_files:
                html_content += f'<img src="{img_file}" alt="{img_file}"><br><br>'
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        output_path = os.path.join(self.viz_dir, f"{os.path.basename(self.results_dir)}_report.html")
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path 