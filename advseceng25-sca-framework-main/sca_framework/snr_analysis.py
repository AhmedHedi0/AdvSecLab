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
from sca_framework import TraceData, PowerModel, HammingWeightEncModel
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class IdentityModel(PowerModel):
    """Identity power model (returns input value directly)."""
    def compute(self, data: int, key_byte: int) -> float:
        return float(data ^ key_byte)

class SNRAnalyzer:
    def __init__(self, power_model: PowerModel = HammingWeightEncModel()):
        self.power_model = power_model
        self.leakage_threshold = 1.0

    def calculate_snr_empirical(self, traces: List[TraceData], correct_key: int, 
                               byte_position: int = 0) -> np.ndarray:
        """Calculate empirical SNR using known correct key."""
        if not traces: raise ValueError("No traces provided")
        max_traces = min(5000, len(traces))
        traces = traces[:max_traces]
        waves = np.array([trace.wave for trace in traces])
        plaintexts = np.array([trace.plaintext for trace in traces])
        plaintext_bytes = np.array([
            (pt >> (8 * (15 - byte_position))) & 0xFF for pt in plaintexts
        ])
        key_bytes = correct_key.to_bytes(16, 'big')
        correct_key_byte = key_bytes[byte_position]
        num_samples = waves.shape[1]
        snr_values = np.zeros(num_samples)
        hyp_power = np.array([
            self.power_model.compute(pt_byte, correct_key_byte)
            for pt_byte in plaintext_bytes
        ])
        for sample_idx in range(num_samples):
            sample_values = waves[:, sample_idx]
            unique_powers = np.unique(hyp_power)
            group_means, group_vars = [], []
            for power in unique_powers:
                mask = hyp_power == power
                group_values = sample_values[mask]
                if len(group_values) > 1:
                    group_means.append(np.mean(group_values))
                    group_vars.append(np.var(group_values))
            if len(group_means) > 1:
                signal_var = np.var(group_means)
                noise_var = np.mean(group_vars)
                if noise_var > 0:
                    snr_values[sample_idx] = signal_var / noise_var
        return snr_values

    def find_leakage_points(self, snr_values: np.ndarray) -> List[int]:
        """Identify points with significant leakage."""
        threshold = np.mean(snr_values) + 2 * np.std(snr_values)
        return np.where(snr_values > threshold)[0].tolist()

    def estimate_attack_complexity(self, snr_values: np.ndarray) -> Dict[str, float]:
        """Estimate attack complexity based on SNR values."""
        max_snr = np.max(snr_values)
        mean_snr = np.mean(snr_values[snr_values > 0])
        cpa_traces = max(100, int(10000 / max_snr)) if max_snr > 0 else float('inf')
        dpa_traces = max(500, int(50000 / max_snr)) if max_snr > 0 else float('inf')
        feasibility = 'High' if max_snr > 1.0 else 'Medium' if max_snr > 0.1 else 'Low'
        return {
            'max_snr': max_snr,
            'mean_snr': mean_snr,
            'estimated_cpa_traces': cpa_traces,
            'estimated_dpa_traces': dpa_traces,
            'attack_feasibility': feasibility
        }

class SNRVisualizer:
    """Handles visualization of SNR analysis results."""
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_snr_trace(self, snr_values: np.ndarray, title: str = "Signal-to-Noise Ratio") -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(snr_values, 'b-', linewidth=1.5)
        plt.xlabel('Sample Index')
        plt.ylabel('SNR')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def demonstrate_snr_analysis(traces: List[TraceData], correct_key: int, byte_index: int = 0):
    """Demonstrate SNR analysis workflow with different power models."""
    visualizer = SNRVisualizer()
    analyzer = SNRAnalyzer()
    snr_empirical = analyzer.calculate_snr_empirical(traces, correct_key, byte_index)
    visualizer.plot_snr_trace(snr_empirical)
    complexity = analyzer.estimate_attack_complexity(snr_empirical)
    print(f"Max SNR: {complexity['max_snr']:.3f}")
    print(f"Estimated CPA traces: {complexity['estimated_cpa_traces']}")

if __name__ == "__main__":
    from sca_framework import TraceCapture
    capture = TraceCapture()
    traces = capture.load_traces(r"C:\Users\Ahmed\Desktop\AdvSecLab\advseceng25-sca-framework-main\src\py\data\traces_1_enhanced.npz")
    demonstrate_snr_analysis(traces, correct_key=0x2b7e151628aed2a6abf7158809cf4f3c, byte_index=0)