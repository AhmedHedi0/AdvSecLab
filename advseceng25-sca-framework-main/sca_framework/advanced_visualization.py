"""
Advanced Visualization Module for Side-Channel Analysis
======================================================

This module provides comprehensive visualization capabilities for side-channel analysis,
including SNR plots, correlation traces, attack progress tracking, and result analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional, Any
from sca_framework import TraceData, AttackResult
from snr_analysis import SNRAnalyzer
import logging

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """Advanced visualization tools for side-channel analysis results."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualizer with specified style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17becf',
            'purple': '#9467bd',
            'brown': '#8c564b',
            'pink': '#e377c2',
            'gray': '#7f7f7f'
        }
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def plot_snr_analysis(self, snr_values: np.ndarray, 
                         title: str = "Signal-to-Noise Ratio Analysis",
                         save_path: Optional[str] = None) -> None:
        """
        Create comprehensive SNR analysis plot.
        
        Args:
            snr_values: SNR values for each sample point
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Main SNR trace
        axes[0, 0].plot(snr_values, color=self.colors['primary'], linewidth=2)
        axes[0, 0].set_title('SNR Over Sample Points')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('SNR')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Highlight high SNR regions
        threshold = np.mean(snr_values) + 2 * np.std(snr_values)
        high_snr_mask = snr_values > threshold
        if np.any(high_snr_mask):
            axes[0, 0].fill_between(range(len(snr_values)), 0, snr_values, 
                                   where=high_snr_mask, alpha=0.3, 
                                   color=self.colors['success'], 
                                   label=f'High SNR (>{threshold:.2f})')
            axes[0, 0].legend()
        
        # SNR histogram
        axes[0, 1].hist(snr_values[snr_values > 0], bins=30, alpha=0.7, 
                       color=self.colors['secondary'], edgecolor='black')
        axes[0, 1].set_title('SNR Distribution')
        axes[0, 1].set_xlabel('SNR Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative SNR
        sorted_snr = np.sort(snr_values)[::-1]
        cumulative = np.cumsum(sorted_snr) / np.sum(sorted_snr)
        axes[1, 0].plot(cumulative, color=self.colors['purple'], linewidth=2)
        axes[1, 0].set_title('Cumulative SNR Distribution')
        axes[1, 0].set_xlabel('Sample Rank')
        axes[1, 0].set_ylabel('Cumulative SNR Fraction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # SNR statistics
        stats_text = f"""SNR Statistics:
Max: {np.max(snr_values):.3f}
Mean: {np.mean(snr_values):.3f}
Std: {np.std(snr_values):.3f}
Median: {np.median(snr_values):.3f}
95th percentile: {np.percentile(snr_values, 95):.3f}
Points > 1.0: {np.sum(snr_values > 1.0)}
Points > 0.5: {np.sum(snr_values > 0.5)}"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('SNR Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attack_progress(self, results: List[AttackResult], 
                           correct_key_byte: int,
                           title: str = "Attack Progress Analysis",
                           save_path: Optional[str] = None) -> None:
        """
        Plot attack progress over increasing number of traces.
        
        Args:
            results: List of attack results
            correct_key_byte: The correct key byte value
            title: Plot title
            save_path: Optional path to save the plot
        """
        if not results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        trace_counts = [r.num_traces_used for r in results]
        success_rates = [r.success_rate for r in results]
        recovered_keys = [r.recovered_key[0] for r in results]
        attack_type = results[0].attack_type
        
        # Success rate over traces
        axes[0, 0].plot(trace_counts, success_rates, 'o-', 
                       color=self.colors['primary'], linewidth=2, markersize=6)
        axes[0, 0].set_title(f'{attack_type} Success Rate vs Traces')
        axes[0, 0].set_xlabel('Number of Traces')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Key recovery accuracy
        correct_recoveries = [1 if key == correct_key_byte else 0 for key in recovered_keys]
        axes[0, 1].plot(trace_counts, correct_recoveries, 'o-', 
                       color=self.colors['success'], linewidth=2, markersize=6)
        axes[0, 1].set_title('Key Recovery Accuracy')
        axes[0, 1].set_xlabel('Number of Traces')
        axes[0, 1].set_ylabel('Correct Recovery (1=Yes, 0=No)')
        axes[0, 1].set_ylim(-0.1, 1.1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Key hypothesis ranking over traces
        if len(results) > 0 and hasattr(results[0], 'key_rankings'):
            final_rankings = results[-1].key_rankings
            top_keys = np.argsort(final_rankings)[-10:][::-1]  # Top 10 keys
            
            for i, key in enumerate(top_keys):
                key_progress = [r.key_rankings[key] for r in results]
                color = self.colors['danger'] if key == correct_key_byte else self.colors['gray']
                alpha = 1.0 if key == correct_key_byte else 0.6
                linewidth = 3 if key == correct_key_byte else 1
                label = f'Key 0x{key:02x}' + (' (Correct)' if key == correct_key_byte else '')
                
                axes[1, 0].plot(trace_counts, key_progress, 'o-', 
                               color=color, alpha=alpha, linewidth=linewidth,
                               markersize=4, label=label)
            
            axes[1, 0].set_title('Key Hypothesis Rankings')
            axes[1, 0].set_xlabel('Number of Traces')
            axes[1, 0].set_ylabel('Ranking Score')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Attack summary statistics
        final_result = results[-1]
        summary_text = f"""Attack Summary:
Algorithm: {attack_type}
Total Traces: {final_result.num_traces_used}
Final Success Rate: {final_result.success_rate:.4f}
Recovered Key: 0x{final_result.recovered_key[0]:02x}
Correct Key: 0x{correct_key_byte:02x}
Success: {'YES' if final_result.recovered_key[0] == correct_key_byte else 'NO'}

Performance:
Best Success Rate: {max(success_rates):.4f}
Traces for Success: {trace_counts[success_rates.index(max(success_rates))]}
Convergence: {'Good' if success_rates[-1] > 0.5 else 'Poor'}"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Attack Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_heatmap(self, correlation_traces: np.ndarray,
                                title: str = "Correlation Heatmap",
                                save_path: Optional[str] = None) -> None:
        """
        Plot correlation traces as a heatmap.
        
        Args:
            correlation_traces: 2D array of correlation values (keys x samples)
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Main heatmap
        im = ax1.imshow(correlation_traces, aspect='auto', cmap='RdBu_r', 
                       interpolation='nearest', vmin=-1, vmax=1)
        ax1.set_title('Correlation Traces for All Key Hypotheses')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Key Hypothesis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Correlation Coefficient')
        
        # Find and highlight maximum correlation
        max_corr_key, max_corr_sample = np.unravel_index(
            np.argmax(np.abs(correlation_traces)), correlation_traces.shape)
        ax1.plot(max_corr_sample, max_corr_key, 'yo', markersize=10, 
                markeredgecolor='black', markeredgewidth=2,
                label=f'Max: Key 0x{max_corr_key:02x}, Sample {max_corr_sample}')
        ax1.legend()
        
        # Plot correlation traces for top candidates
        max_correlations = np.max(np.abs(correlation_traces), axis=1)
        top_keys = np.argsort(max_correlations)[-5:][::-1]  # Top 5 keys
        
        for i, key in enumerate(top_keys):
            color = self.colors['danger'] if i == 0 else plt.cm.tab10(i)
            ax2.plot(correlation_traces[key], color=color, linewidth=2,
                    label=f'Key 0x{key:02x} (max: {max_correlations[key]:.3f})')
        
        ax2.set_title('Top 5 Correlation Traces')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attack_comparison(self, cpa_results: List[AttackResult], 
                              dpa_results: List[AttackResult],
                              correct_key_byte: int,
                              title: str = "CPA vs DPA Comparison",
                              save_path: Optional[str] = None) -> None:
        """
        Compare CPA and DPA attack results.
        
        Args:
            cpa_results: CPA attack results
            dpa_results: DPA attack results
            correct_key_byte: The correct key byte value
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Success rates comparison
        if cpa_results:
            cpa_traces = [r.num_traces_used for r in cpa_results]
            cpa_success = [r.success_rate for r in cpa_results]
            axes[0, 0].plot(cpa_traces, cpa_success, 'o-', 
                           color=self.colors['primary'], linewidth=2, 
                           label='CPA', markersize=6)
        
        if dpa_results:
            dpa_traces = [r.num_traces_used for r in dpa_results]
            dpa_success = [r.success_rate for r in dpa_results]
            axes[0, 0].plot(dpa_traces, dpa_success, 's-', 
                           color=self.colors['secondary'], linewidth=2, 
                           label='DPA', markersize=6)
        
        axes[0, 0].set_title('Success Rate Comparison')
        axes[0, 0].set_xlabel('Number of Traces')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Key recovery accuracy
        if cpa_results:
            cpa_accuracy = [1 if r.recovered_key[0] == correct_key_byte else 0 
                           for r in cpa_results]
            axes[0, 1].plot(cpa_traces, cpa_accuracy, 'o-', 
                           color=self.colors['primary'], linewidth=2, 
                           label='CPA', markersize=6)
        
        if dpa_results:
            dpa_accuracy = [1 if r.recovered_key[0] == correct_key_byte else 0 
                           for r in dpa_results]
            axes[0, 1].plot(dpa_traces, dpa_accuracy, 's-', 
                           color=self.colors['secondary'], linewidth=2, 
                           label='DPA', markersize=6)
        
        axes[0, 1].set_title('Key Recovery Accuracy')
        axes[0, 1].set_xlabel('Number of Traces')
        axes[0, 1].set_ylabel('Correct Recovery')
        axes[0, 1].set_ylim(-0.1, 1.1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance metrics
        metrics = ['Algorithm', 'Max Success Rate', 'Min Traces for Success', 
                  'Final Accuracy', 'Convergence']
        
        cpa_metrics = ['CPA', 
                      f"{max(cpa_success):.3f}" if cpa_results else "N/A",
                      f"{cpa_traces[cpa_success.index(max(cpa_success))]}" if cpa_results else "N/A",
                      f"{'YES' if cpa_results and cpa_results[-1].recovered_key[0] == correct_key_byte else 'NO'}",
                      f"{'Good' if cpa_results and cpa_success[-1] > 0.5 else 'Poor'}"]
        
        dpa_metrics = ['DPA',
                      f"{max(dpa_success):.3f}" if dpa_results else "N/A", 
                      f"{dpa_traces[dpa_success.index(max(dpa_success))]}" if dpa_results else "N/A",
                      f"{'YES' if dpa_results and dpa_results[-1].recovered_key[0] == correct_key_byte else 'NO'}",
                      f"{'Good' if dpa_results and dpa_success[-1] > 0.1 else 'Poor'}"]
        
        # Create comparison table
        table_data = list(zip(metrics, cpa_metrics, dpa_metrics))
        table = axes[1, 0].table(cellText=table_data, 
                                colLabels=['Metric', 'CPA', 'DPA'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Performance Comparison')
        
        # Efficiency analysis
        if cpa_results and dpa_results:
            efficiency_text = f"""Efficiency Analysis:
            
CPA Performance:
- Fastest success: {min(cpa_traces)} traces
- Best correlation: {max(cpa_success):.4f}
- Consistency: {'High' if np.std(cpa_success[-3:]) < 0.1 else 'Low'}

DPA Performance:  
- Fastest success: {min(dpa_traces)} traces
- Best ranking: {max(dpa_success):.4f}
- Consistency: {'High' if np.std(dpa_success[-3:]) < 0.01 else 'Low'}

Recommendation:
{'CPA is more efficient' if max(cpa_success) > max(dpa_success) else 'DPA is more efficient'}
for this target and trace quality."""
        else:
            efficiency_text = "Insufficient data for efficiency analysis"
        
        axes[1, 1].text(0.1, 0.9, efficiency_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Efficiency Analysis')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def demonstrate_visualizations():
    """Demonstrate advanced visualization capabilities."""
    print("Advanced Visualization Demonstration")
    print("=" * 35)
    
    # Load previous results
    from sca_framework import TraceCapture
    from attack_algorithms import CorrelationPowerAnalysis, DifferentialPowerAnalysis, AttackConfig, HammingWeightModel
    
    capture = TraceCapture()
    
    try:
        traces = capture.load_traces(r"C:\Users\Ahmed\Desktop\AdvSecLab\advseceng25-sca-framework-main\src\py\data\traces_1_filtered.npz")
        print(f"Loaded {len(traces)} traces for visualization")
    except Exception as e:
        print(f"Error loading traces: {e}")
        return
    
    # Initialize visualizer
    visualizer = AdvancedVisualizer()
    
    # Known key from simulation
    correct_key = 0x2b7e151628aed2a6abf7158809cf4f3c
    correct_key_bytes = list(correct_key.to_bytes(16, 'big'))
    
    # Run quick attacks for visualization
    print("Running attacks for visualization...")
    
    # CPA attack
    cpa_config = AttackConfig(
        power_model=HammingWeightModel(),
        target_byte=0,
        incremental_size=100,
        report_interval=200,
        max_traces=300
    )
    cpa_attacker = CorrelationPowerAnalysis(cpa_config)
    cpa_results = cpa_attacker.incremental_attack(traces[:300], 0)
    
    # DPA attack  
    dpa_config = AttackConfig(
        power_model=HammingWeightModel(),
        target_byte=0,
        incremental_size=100,
        report_interval=200,
        max_traces=300
    )
    dpa_attacker = DifferentialPowerAnalysis(dpa_config)
    dpa_results = dpa_attacker.incremental_attack(traces[:300], 0)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Attack progress plots
    if cpa_results:
        visualizer.plot_attack_progress(cpa_results, correct_key_bytes[0], 
                                       "CPA Attack Progress")
    
    if dpa_results:
        visualizer.plot_attack_progress(dpa_results, correct_key_bytes[0], 
                                       "DPA Attack Progress")
    
    # Comparison plot
    if cpa_results and dpa_results:
        visualizer.plot_attack_comparison(cpa_results, dpa_results, 
                                         correct_key_bytes[0])
    
    # Correlation heatmap
    if cpa_results:
        final_correlations = cpa_results[-1].correlation_traces
        visualizer.plot_correlation_heatmap(final_correlations, 
                                           "CPA Correlation Heatmap")
    
    print("Visualization demonstration completed!")

if __name__ == "__main__":
    demonstrate_visualizations()

