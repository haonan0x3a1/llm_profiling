import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import textwrap
from typing import Dict, Tuple, Optional
import warnings

# Suppress matplotlib and seaborn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# 0. Global Setup & Data Preparation
# ==============================================================================

class PlotConfig:
    """Configuration class for unified plot parameter management"""
    def __init__(self):
        self.output_dir = 'Result_Figures'
        self.dpi = 300
        self.figure_size = (12, 8)
        
        # Academic color schemes
        self.colors = {
            'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            'secondary': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
            'kernel': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
            'api': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
        
        # Professional font configuration
        self.font_config = {
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'legend.title_fontsize': 13,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }

def setup_environment(config: PlotConfig) -> str:
    """Create output directory and setup global plotting style"""
    try:
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
            print(f"Created output directory: {config.output_dir}")
        
        # Set academic style
        sns.set_style("whitegrid")
        plt.rcParams.update(config.font_config)
        
        return config.output_dir
    except Exception as e:
        print(f"Environment setup failed: {e}")
        raise

def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare all DataFrames for plotting with data validation"""
    try:
        # Performance metrics data
        perf_data = {
            'Model': ['Qwen2.5-72B-Instruct'] * 3 + ['DeepSeek-R1-Distill-Llama-70B'] * 3,
            'Requests': [1, 1, 5, 1, 1, 5],
            'Max Tokens': [256, 512, 512, 256, 512, 512],
            'Scenario': ['1 Req / 256 Tokens', '1 Req / 512 Tokens', '5 Reqs / 512 Tokens'] * 2,
            'Throughput (tokens/s)': [38.75, 38.32, 38.32, 38.73, 38.72, 38.52],
            'Latency (ms/token)': [25.81, 26.09, 26.10, 25.82, 25.83, 25.96]
        }
        perf_df = pd.DataFrame(perf_data)

        # NSYS GPU kernel time distribution
        nsys_kernel_data = {
            'Model': ['Qwen2.5-72B-Instruct', 'DeepSeek-R1-Distill-Llama-70B'],
            'NCCL AllReduce (%)': [75.6, 83.1],
            'GEMM Computation (%)': [14.0, 8.2],
            'Flash Attention (%)': [1.3, 0.6],
        }
        nsys_kernel_df = pd.DataFrame(nsys_kernel_data)
        nsys_kernel_df['Others (%)'] = 100 - nsys_kernel_df.iloc[:, 1:].sum(axis=1)

        # NSYS CUDA API call total time
        cuda_api_data = {
            'API Call': ['cudaStreamSynchronize', 'cudaGraphInstantiate', 
                        'cudaMemcpyAsync', 'cuLaunchKernel', 'cudaDeviceSynchronize'],
            'Qwen2.5-72B-Instruct (s)': [84.4, 85.3, 43.5, 2.4, 35.0],
            'DeepSeek-R1-Distill-Llama-70B (s)': [104.2, 67.1, 51.5, 28.9, 24.3]
        }
        cuda_api_df = pd.DataFrame(cuda_api_data)

        # Environment configuration
        env_data = {
            'Category': ['Hardware'] * 3 + ['Software'] * 7,
            'Component': ['GPU', 'GPU Count', 'Interconnect', 'PyTorch', 
                         'CUDA Toolkit (Driver)', 'vLLM', 'Transformers', 
                         'NCCL', 'Triton', 'xFormers'],
            'Version / Specification': ['NVIDIA A800 80GB', '8', 'PCIe', '2.6.0', 
                                      '12.4', '0.8.3', '4.51.1', '2.21.5', 
                                      '3.2.0', '0.0.29.post2']
        }
        env_df = pd.DataFrame(env_data)
        
        # Data validation
        for name, df in [('perf', perf_df), ('nsys_kernel', nsys_kernel_df), 
                        ('cuda_api', cuda_api_df), ('env', env_df)]:
            if df.empty:
                raise ValueError(f"{name} DataFrame is empty")
        
        return perf_df, nsys_kernel_df, cuda_api_df, env_df
        
    except Exception as e:
        print(f"Data preparation failed: {e}")
        raise

# ==============================================================================
# 2. Plotting Functions
# ==============================================================================

def save_table_as_image(df: pd.DataFrame, filepath: str, title: str, 
                       config: PlotConfig) -> None:
    """Save pandas DataFrame as styled image"""
    try:
        df_wrapped = df.copy()
        
        # Special handling for model names - shorten them
        if 'Model' in df_wrapped.columns:
            df_wrapped['Model'] = df_wrapped['Model'].apply(
                lambda x: x.replace('DeepSeek-R1-Distill-Llama-70B', 'DeepSeek-R1-70B')
                          .replace('Qwen2.5-72B-Instruct', 'Qwen2.5-72B')
            )
        
        # Text wrapping for other columns
        for col in df_wrapped.columns:
            if df_wrapped[col].dtype == 'object' and col != 'Model':
                df_wrapped[col] = df_wrapped[col].apply(
                    lambda x: '\n'.join(textwrap.wrap(str(x), width=25))
                )

        fig_height = max(4, 0.6 * len(df_wrapped) + 2)
        fig, ax = plt.subplots(figsize=(15, fig_height))
        ax.axis('off')
        
        table = ax.table(
            cellText=df_wrapped.values, 
            colLabels=df_wrapped.columns, 
            cellLoc='center', 
            loc='center',
            colColours=["#f8f9fa"] * len(df_wrapped.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.3, 2.4)

        # Professional table styling
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('#dee2e6')
            cell.set_linewidth(1)
            if key[0] == 0:  # Header row
                cell.set_text_props(weight='bold', color='#212529')
                cell.set_facecolor('#e9ecef')
            else:
                cell.set_text_props(color='#495057')
                cell.set_facecolor('#ffffff' if key[0] % 2 == 0 else '#f8f9fa')
        
        plt.title(title, fontsize=18, weight='bold', pad=30, color='#212529')
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight', 
                   pad_inches=0.3, facecolor='white', edgecolor='none')
        print(f"Generated figure: {filepath}")
        
    except Exception as e:
        print(f"Table image generation failed {filepath}: {e}")
    finally:
        plt.close()

def plot_performance_bars(df: pd.DataFrame, y_col: str, y_label: str, 
                         title: str, filepath: str, config: PlotConfig,
                         y_limit: Optional[float] = None) -> None:
    """Generic bar chart plotting function with academic styling"""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Shorten model names for better display
        df_plot = df.copy()
        df_plot['Model'] = df_plot['Model'].apply(
            lambda x: x.replace('DeepSeek-R1-Distill-Llama-70B', 'DeepSeek-R1-70B')
                      .replace('Qwen2.5-72B-Instruct', 'Qwen2.5-72B')
        )
        
        # Create bar plot with custom colors
        bars = sns.barplot(data=df_plot, x='Scenario', y=y_col, hue='Model', 
                          palette=config.colors['primary'][:2], alpha=0.85, ax=ax)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=4, fontsize=11, 
                        weight='bold', color='#2c3e50')
        
        # Professional styling
        ax.set_title(title, weight='bold', fontsize=16, pad=25, color='#2c3e50')
        ax.set_ylabel(y_label, fontsize=14, color='#34495e', weight='bold')
        ax.set_xlabel('Test Scenarios (Requests, Max Output Tokens)', 
                     fontsize=14, color='#34495e', weight='bold')
        
        if y_limit:
            ax.set_ylim(0, y_limit)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        
        # Legend styling - position to avoid blocking data
        if 'latency' in filepath.lower():
            legend_loc = 'lower right'
        else:
            legend_loc = 'upper left'
            
        legend = ax.legend(title='Model', frameon=True, fancybox=True, 
                          shadow=True, loc=legend_loc)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_alpha(0.9)
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=config.dpi, facecolor='white', 
                   bbox_inches='tight', edgecolor='none')
        print(f"Generated figure: {filepath}")
        
    except Exception as e:
        print(f"Performance bar chart generation failed {filepath}: {e}")
    finally:
        plt.close()

def plot_batch_size_impact(df: pd.DataFrame, output_dir: str, config: PlotConfig) -> None:
    """Plot performance trend with batch size impact - split into two subplots"""
    try:
        df_subset = df[df['Max Tokens'] == 512].copy()
        if df_subset.empty:
            print("Warning: No data found for Max Tokens = 512")
            return
        
        # Shorten model names
        df_subset['Model_Short'] = df_subset['Model'].apply(
            lambda x: x.replace('DeepSeek-R1-Distill-Llama-70B', 'DeepSeek-R1-70B')
                      .replace('Qwen2.5-72B-Instruct', 'Qwen2.5-72B')
        )
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        models = df_subset['Model'].unique()
        colors = config.colors['primary'][:len(models)]
        
        # Subplot 1: Throughput
        for i, model in enumerate(models):
            model_data = df_subset[df_subset['Model'] == model]
            model_short = model_data['Model_Short'].iloc[0]
            ax1.plot(model_data['Requests'], model_data['Throughput (tokens/s)'], 
                    marker='o', linewidth=3, markersize=10, label=model_short,
                    color=colors[i], alpha=0.8)

        ax1.set_xlabel('Concurrent Requests (Batch Size)', fontsize=14, weight='bold')
        ax1.set_ylabel('Throughput (tokens/s)', fontsize=14, weight='bold')
        ax1.set_title('Throughput vs Batch Size', fontsize=14, weight='bold', pad=20)
        ax1.set_xticks(df_subset['Requests'].unique())
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add value labels for throughput
        for i, model in enumerate(models):
            model_data = df_subset[df_subset['Model'] == model]
            for _, row in model_data.iterrows():
                ax1.annotate(f'{row["Throughput (tokens/s)"]:.1f}', 
                           (row['Requests'], row['Throughput (tokens/s)']),
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=9, weight='bold')

        # Subplot 2: Latency
        for i, model in enumerate(models):
            model_data = df_subset[df_subset['Model'] == model]
            model_short = model_data['Model_Short'].iloc[0]
            ax2.plot(model_data['Requests'], model_data['Latency (ms/token)'], 
                    marker='s', linewidth=3, markersize=10, label=model_short,
                    color=colors[i], alpha=0.8)

        ax2.set_xlabel('Concurrent Requests (Batch Size)', fontsize=14, weight='bold')
        ax2.set_ylabel('Latency (ms/token)', fontsize=14, weight='bold')
        ax2.set_title('Latency vs Batch Size', fontsize=14, weight='bold', pad=20)
        ax2.set_xticks(df_subset['Requests'].unique())
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add value labels for latency
        for i, model in enumerate(models):
            model_data = df_subset[df_subset['Model'] == model]
            for _, row in model_data.iterrows():
                ax2.annotate(f'{row["Latency (ms/token)"]:.1f}', 
                           (row['Requests'], row['Latency (ms/token)']),
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=9, weight='bold')

        # Overall title
        fig.suptitle('Impact of Batch Size on Performance', fontsize=16, weight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, "fig3_batch_size_impact.png")
        plt.savefig(filepath, dpi=config.dpi, facecolor='white', 
                   bbox_inches='tight', edgecolor='none')
        print(f"Generated figure: {filepath}")
        
    except Exception as e:
        print(f"Batch size impact plot generation failed: {e}")
    finally:
        plt.close()

def plot_kernel_distribution_stacked_bar(df: pd.DataFrame, output_dir: str, 
                                        config: PlotConfig) -> None:
    """Plot stacked bar chart for GPU kernel time distribution"""
    try:
        df_plot = df.set_index('Model').sort_values(by='NCCL AllReduce (%)', ascending=True)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use custom colors for better academic appearance
        bars = df_plot.plot(kind='barh', stacked=True, ax=ax, 
                           color=config.colors['kernel'], width=0.6, alpha=0.85)
        
        ax.set_title('GPU Kernel Time Distribution (1 Request / 256 Tokens)', 
                    weight='bold', fontsize=16, pad=25, color='#2c3e50')
        ax.set_xlabel('Percentage of Total GPU Time (%)', fontsize=14, 
                     color='#34495e', weight='bold')
        ax.set_ylabel('')
        
        # Legend styling
        legend = ax.legend(title='Kernel Type', bbox_to_anchor=(1.02, 1), 
                          loc='upper left', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_alpha(0.9)
        
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)

        # Add percentage labels
        for c in ax.containers:
            labels = [f'{v:.1f}%' if v > 5 else '' for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='center', 
                        color='white', weight='bold', fontsize=10)
            
        plt.tight_layout()
        filepath = os.path.join(output_dir, "fig5_kernel_distribution_bar.png")
        plt.savefig(filepath, dpi=config.dpi, facecolor='white', 
                   bbox_inches='tight', edgecolor='none')
        print(f"Generated figure: {filepath}")
        
    except Exception as e:
        print(f"Kernel distribution plot generation failed: {e}")
    finally:
        plt.close()

def plot_cuda_api_comparison(df: pd.DataFrame, output_dir: str, 
                           config: PlotConfig) -> None:
    """Plot grouped bar chart for CUDA API call time comparison"""
    try:
        df_melted = df.melt(id_vars='API Call', var_name='Model', value_name='Time (s)')
        
        # Simplify model names for better visualization
        df_melted['Model'] = df_melted['Model'].str.replace(' (s)', '').str.split('-').str[0]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = sns.barplot(data=df_melted, x='API Call', y='Time (s)', 
                          hue='Model', palette=config.colors['api'][:2], 
                          alpha=0.85, ax=ax)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=4, fontsize=10, 
                        weight='bold', color='#2c3e50')
        
        ax.set_title('CUDA API Call Time Consumption (CPU-side Overhead)', 
                    weight='bold', fontsize=16, pad=25, color='#2c3e50')
        ax.set_ylabel('Total Time (seconds)', fontsize=14, color='#34495e', weight='bold')
        ax.set_xlabel('CUDA API Calls', fontsize=14, color='#34495e', weight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        
        # Legend styling
        legend = ax.legend(title='Model', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_alpha(0.9)
        
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "fig6_cuda_api_comparison.png")
        plt.savefig(filepath, dpi=config.dpi, facecolor='white', 
                   bbox_inches='tight', edgecolor='none')
        print(f"Generated figure: {filepath}")
        
    except Exception as e:
        print(f"CUDA API comparison plot generation failed: {e}")
    finally:
        plt.close()

# ==============================================================================
# 3. Main Execution
# ==============================================================================
def main():
    """Main function to coordinate all plotting operations"""
    try:
        print("Starting LLM Performance Analysis Report Figure Generation...")
        
        # Initialize configuration
        config = PlotConfig()
        output_dir = setup_environment(config)
        
        # Prepare data
        print("Preparing data...")
        perf_df, nsys_kernel_df, cuda_api_df, env_df = prepare_data()
        
        # Generate tables
        print("Generating configuration tables...")
        save_table_as_image(env_df, os.path.join(output_dir, 'fig0_environment_config.png'), 
                           'Experimental Environment Configuration', config)
        save_table_as_image(perf_df, os.path.join(output_dir, 'fig4_performance_summary.png'), 
                           'Performance Metrics Summary', config)

        # Generate performance charts
        print("Generating performance charts...")
        plot_performance_bars(perf_df, 'Throughput (tokens/s)', 'Throughput (tokens/s)',
                             'Model Inference Throughput (8x A800, TP=8)',
                             os.path.join(output_dir, 'fig1_throughput_comparison.png'), 
                             config, y_limit=50)
        
        plot_performance_bars(perf_df, 'Latency (ms/token)', 'Latency (ms/token)',
                             'Per-Token Generation Latency (8x A800, TP=8)',
                             os.path.join(output_dir, 'fig2_latency_comparison.png'), 
                             config, y_limit=30)
        
        plot_batch_size_impact(perf_df, output_dir, config)

        # Generate NSYS analysis charts
        print("Generating NSYS analysis charts...")
        plot_kernel_distribution_stacked_bar(nsys_kernel_df, output_dir, config)
        plot_cuda_api_comparison(cuda_api_df, output_dir, config)

        print(f"\n✅ All figures have been successfully generated in '{output_dir}' directory")
        
    except Exception as e:
        print(f"❌ Program execution failed: {e}")
        raise

if __name__ == '__main__':
    main()