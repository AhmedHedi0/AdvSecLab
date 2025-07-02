"""
Signal-to-Noise Ratio (SNR) Analysis Module
==========================================

This module implements SNR calculation functions for power analysis traces.
SNR is a critical metric for assessing the quality of side-channel measurements
and predicting attack success rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from sca_framework import TraceData, PowerModel, HammingWeightModel
import logging

logger = logging.getLogger(__name__)

class SNRAnalyzer:
    """
    Signal-to-Noise Ratio analyzer for power traces.
    
    The SNR is calculated as the ratio of signal variance to noise variance:
    SNR = Var(Signal) / Var(Noise)
    
    For side-channel analysis, the signal is the power consumption that depends
    on the secret key, while noise is everything else.
    """
    
    def __init__(self, power_model: PowerModel = None):
        self.power_model = power_model or HammingWeightModel()
        self.snr_cache = {}
    
    def calculate_snr_theoretical(self, traces: List[TraceData], 
                                 byte_position: int = 0) -> np.ndarray:
        """
        Calculate theoretical SNR based on power model predictions.
        
        Args:
            traces: List of power traces
            byte_position: Which byte of the key to analyze (0-15)
            
        Returns:
            SNR values for each sample point
        """
        if not traces:
            raise ValueError("No traces provided")
        
        # Extract data
        waves = np.array([trace.wave for trace in traces])
        plaintexts = np.array([trace.plaintext for trace in traces])
        
        # Extract the relevant byte from plaintexts
        plaintext_bytes = []
        for pt in plaintexts:
            pt_bytes = list(pt.to_bytes(16, 'big'))
            plaintext_bytes.append(pt_bytes[byte_position])
        plaintext_bytes = np.array(plaintext_bytes)
        
        num_samples = waves.shape[1]
        snr_values = np.zeros(num_samples)
        
        # Calculate SNR for each sample point
        for sample_idx in range(num_samples):
            sample_values = waves[:, sample_idx]
            
            # Group traces by hypothetical power consumption
            power_groups = {}
            for i, pt_byte in enumerate(plaintext_bytes):
                # Calculate hypothetical power for all possible key bytes
                for key_guess in range(256):
                    hyp_power = self.power_model.compute(pt_byte, key_guess)
                    if hyp_power not in power_groups:
                        power_groups[hyp_power] = []
                    power_groups[hyp_power].append(sample_values[i])
            
            # Calculate signal and noise variance
            signal_variance = self._calculate_signal_variance(power_groups)
            noise_variance = self._calculate_noise_variance(power_groups)
            
            if noise_variance > 0:
                snr_values[sample_idx] = signal_variance / noise_variance
            else:
                snr_values[sample_idx] = 0
        
        return snr_values
    
    def calculate_snr_empirical(self, traces: List[TraceData], 
                               correct_key: int, byte_position: int = 0) -> np.ndarray:
        """
        Calculate empirical SNR using the known correct key.
        
        Args:
            traces: List of power traces
            correct_key: The correct key value
            byte_position: Which byte of the key to analyze (0-15)
            
        Returns:
            SNR values for each sample point
        """
        if not traces:
            raise ValueError("No traces provided")
        
        # Extract data
        waves = np.array([trace.wave for trace in traces])
        plaintexts = np.array([trace.plaintext for trace in traces])
        
        # Extract the relevant bytes
        plaintext_bytes = []
        for pt in plaintexts:
            pt_bytes = list(pt.to_bytes(16, 'big'))
            plaintext_bytes.append(pt_bytes[byte_position])
        plaintext_bytes = np.array(plaintext_bytes)
        
        key_bytes = list(correct_key.to_bytes(16, 'big'))
        correct_key_byte = key_bytes[byte_position]
        
        num_samples = waves.shape[1]
        snr_values = np.zeros(num_samples)
        
        # Calculate hypothetical power consumption for each trace
        hyp_power = np.array([
            self.power_model.compute(pt_byte, correct_key_byte) 
            for pt_byte in plaintext_bytes
        ])
        
        # Calculate SNR for each sample point
        for sample_idx in range(num_samples):
            sample_values = waves[:, sample_idx]
            
            # Group by hypothetical power consumption
            unique_powers = np.unique(hyp_power)
            group_means = []
            group_vars = []
            
            for power in unique_powers:
                mask = hyp_power == power
                if np.sum(mask) > 1:  # Need at least 2 samples for variance
                    group_values = sample_values[mask]
                    group_means.append(np.mean(group_values))
                    group_vars.append(np.var(group_values))
            
            if len(group_means) > 1:
                # Signal variance: variance of group means
                signal_var = np.var(group_means)
                # Noise variance: average of within-group variances
                noise_var = np.mean(group_vars)
                
                if noise_var > 0:
                    snr_values[sample_idx] = signal_var / noise_var
        
        return snr_values
    
    def _calculate_signal_variance(self, power_groups: Dict) -> float:
        """Calculate signal variance from power groups."""
        group_means = []
        for power_level, values in power_groups.items():
            if len(values) > 0:
                group_means.append(np.mean(values))
        
        if len(group_means) > 1:
            return np.var(group_means)
        return 0.0
    
    def _calculate_noise_variance(self, power_groups: Dict) -> float:
        """Calculate noise variance from power groups."""
        group_variances = []
        for power_level, values in power_groups.items():
            if len(values) > 1:  # Need at least 2 values for variance
                group_variances.append(np.var(values))
        
        if len(group_variances) > 0:
            return np.mean(group_variances)
        return 0.0
    
    def calculate_snr_multivariate(self, traces: List[TraceData], 
                                  correct_key: int) -> np.ndarray:
        """
        Calculate SNR for all 16 key bytes simultaneously.
        
        Args:
            traces: List of power traces
            correct_key: The correct key value
            
        Returns:
            SNR matrix of shape (16, num_samples)
        """
        if not traces:
            raise ValueError("No traces provided")
        
        num_samples = len(traces[0].wave)
        snr_matrix = np.zeros((16, num_samples))
        
        for byte_pos in range(16):
            logger.info(f"Calculating SNR for key byte {byte_pos}")
            snr_matrix[byte_pos] = self.calculate_snr_empirical(
                traces, correct_key, byte_pos
            )
        
        return snr_matrix
    
    def find_leakage_points(self, snr_values: np.ndarray, 
                           threshold: float = 1.0) -> List[int]:
        """
        Find sample points with significant leakage (high SNR).
        
        Args:
            snr_values: SNR values for each sample
            threshold: Minimum SNR threshold for leakage detection
            
        Returns:
            List of sample indices with SNR above threshold
        """
        leakage_points = np.where(snr_values > threshold)[0]
        return leakage_points.tolist()
    
    def estimate_attack_complexity(self, snr_values: np.ndarray) -> Dict[str, float]:
        """
        Estimate attack complexity based on SNR values.
        
        Args:
            snr_values: SNR values for each sample
            
        Returns:
            Dictionary with complexity estimates
        """
        max_snr = np.max(snr_values)
        mean_snr = np.mean(snr_values[snr_values > 0])
        
        # Empirical formulas for trace requirements
        # These are rough estimates based on literature
        if max_snr > 0:
            # CPA typically requires fewer traces
            cpa_traces = max(100, int(10000 / max_snr))
            # DPA typically requires more traces
            dpa_traces = max(500, int(50000 / max_snr))
        else:
            cpa_traces = float('inf')
            dpa_traces = float('inf')
        
        return {
            'max_snr': max_snr,
            'mean_snr': mean_snr,
            'estimated_cpa_traces': cpa_traces,
            'estimated_dpa_traces': dpa_traces,
            'attack_feasibility': 'High' if max_snr > 1.0 else 'Medium' if max_snr > 0.1 else 'Low'
        }

class SNRVisualizer:
    """Visualization tools for SNR analysis results."""
    
    def __init__(self):
        plt.style.use('default')
    
    def plot_snr_trace(self, snr_values: np.ndarray, 
                      title: str = "Signal-to-Noise Ratio") -> None:
        """Plot SNR values over sample points."""
        plt.figure(figsize=(12, 6))
        plt.plot(snr_values, 'b-', linewidth=2)
        plt.xlabel('Sample Index')
        plt.ylabel('SNR')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Highlight high SNR regions
        threshold = np.mean(snr_values) + 2 * np.std(snr_values)
        high_snr_points = np.where(snr_values > threshold)[0]
        if len(high_snr_points) > 0:
            plt.scatter(high_snr_points, snr_values[high_snr_points], 
                       color='red', s=30, alpha=0.7, label=f'High SNR (>{threshold:.2f})')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_snr_heatmap(self, snr_matrix: np.ndarray, 
                        title: str = "SNR Heatmap (All Key Bytes)") -> None:
        """Plot SNR heatmap for all key bytes."""
        plt.figure(figsize=(15, 8))
        
        # Create heatmap
        im = plt.imshow(snr_matrix, aspect='auto', cmap='viridis', 
                       interpolation='nearest')
        
        plt.colorbar(im, label='SNR')
        plt.xlabel('Sample Index')
        plt.ylabel('Key Byte Position')
        plt.title(title)
        
        # Add text annotations for high SNR regions
        max_snr_per_byte = np.max(snr_matrix, axis=1)
        for byte_idx, max_snr in enumerate(max_snr_per_byte):
            if max_snr > 1.0:  # Only annotate significant SNR
                max_sample = np.argmax(snr_matrix[byte_idx])
                plt.text(max_sample, byte_idx, f'{max_snr:.1f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_snr_comparison(self, snr_dict: Dict[str, np.ndarray], 
                           title: str = "SNR Comparison") -> None:
        """Compare SNR values from different methods."""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (method, snr_values) in enumerate(snr_dict.items()):
            color = colors[i % len(colors)]
            plt.plot(snr_values, color=color, linewidth=2, 
                    label=method, alpha=0.8)
        
        plt.xlabel('Sample Index')
        plt.ylabel('SNR')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_attack_complexity(self, complexity_results: Dict[str, float]) -> None:
        """Visualize attack complexity estimates."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SNR metrics
        snr_metrics = ['max_snr', 'mean_snr']
        snr_values = [complexity_results[metric] for metric in snr_metrics]
        
        ax1.bar(snr_metrics, snr_values, color=['blue', 'green'], alpha=0.7)
        ax1.set_ylabel('SNR Value')
        ax1.set_title('SNR Metrics')
        ax1.grid(True, alpha=0.3)
        
        # Trace requirements
        attack_methods = ['CPA', 'DPA']
        trace_counts = [
            complexity_results['estimated_cpa_traces'],
            complexity_results['estimated_dpa_traces']
        ]
        
        # Cap at reasonable maximum for visualization
        trace_counts = [min(count, 100000) for count in trace_counts]
        
        ax2.bar(attack_methods, trace_counts, color=['orange', 'red'], alpha=0.7)
        ax2.set_ylabel('Estimated Traces Required')
        ax2.set_title('Attack Complexity')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add feasibility text
        feasibility = complexity_results['attack_feasibility']
        fig.suptitle(f'Attack Feasibility: {feasibility}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def demonstrate_snr_analysis():
    """Demonstrate SNR analysis functionality."""
    print("SNR Analysis Demonstration")
    print("=" * 30)
    
    # Load traces from previous demo
    from sca_framework import TraceCapture
    capture = TraceCapture()
    
    try:
        traces = capture.load_traces("/home/ubuntu/sca_framework/demo_traces.npz")
        print(f"Loaded {len(traces)} traces for SNR analysis")
    except Exception as e:
        print(f"Error loading traces: {e}")
        print("No traces found. Please run demo_trace_capture.py first.")
        return
    
    # Initialize SNR analyzer
    snr_analyzer = SNRAnalyzer()
    visualizer = SNRVisualizer()
    
    # Known key from simulation
    correct_key = 0x2b7e151628aed2a6abf7158809cf4f3c
    
    # Calculate SNR for first key byte
    print("Calculating empirical SNR for key byte 0...")
    try:
        snr_empirical = snr_analyzer.calculate_snr_empirical(traces, correct_key, 0)
        print(f"SNR calculation completed. Max SNR: {np.max(snr_empirical):.3f}")
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return
    
    # Find leakage points
    leakage_points = snr_analyzer.find_leakage_points(snr_empirical, threshold=0.1)
    print(f"Found {len(leakage_points)} leakage points above threshold 0.1")
    
    # Estimate attack complexity
    complexity = snr_analyzer.estimate_attack_complexity(snr_empirical)
    print("\nAttack Complexity Estimates:")
    for key, value in complexity.items():
        print(f"  {key}: {value}")
    
    print("\nSNR analysis demonstration completed successfully!")
    return snr_empirical, complexity

if __name__ == "__main__":
    demonstrate_snr_analysis()

