"""
Side-Channel Attack Algorithms Module
====================================

This module implements Differential Power Analysis (DPA) and Correlation Power Analysis (CPA)
algorithms for recovering secret keys from power traces.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sca_framework import TraceData, PowerModel, AttackResult, HammingWeightEncModel
import logging

logger = logging.getLogger(__name__)

@dataclass
class AttackConfig:
    """Configuration for side-channel attacks."""
    power_model: PowerModel
    target_byte: int = 0
    incremental_size: int = 100
    report_interval: int = 100
    max_traces: Optional[int] = None

class DifferentialPowerAnalysis:
    """Differential Power Analysis (DPA) implementation."""
    
    def __init__(self, config: AttackConfig):
        self.config = config
    
    def attack_single_byte(self, traces: List[TraceData], target_byte: int) -> Tuple[np.ndarray, np.ndarray]:
        if not traces: raise ValueError("No traces provided")
        waves = np.array([trace.wave for trace in traces])
        plaintexts = np.array([trace.plaintext for trace in traces])
        plaintext_bytes = np.array([
            (pt >> (8 * (15 - target_byte))) & 0xFF for pt in plaintexts
        ])
        num_samples = waves.shape[1]
        differential_traces = np.zeros((256, num_samples))
        key_rankings = np.zeros(256)
        for key_guess in range(256):
            selection_bits = np.array([
                (self.config.power_model.compute(pt_byte, key_guess) >> 0) & 1
                for pt_byte in plaintext_bytes
            ])
            set_0 = waves[selection_bits == 0]
            set_1 = waves[selection_bits == 1]
            if len(set_0) > 0 and len(set_1) > 0:
                differential_traces[key_guess] = np.mean(set_1, axis=0) - np.mean(set_0, axis=0)
                key_rankings[key_guess] = np.max(np.abs(differential_traces[key_guess]))
            else:
                key_rankings[key_guess] = 0
        return differential_traces, key_rankings
    
    def incremental_attack(self, traces: List[TraceData], target_byte: int) -> List[AttackResult]:
        results = []
        increment = self.config.incremental_size
        for num_traces in range(increment, len(traces) + 1, increment):
            if self.config.max_traces and num_traces > self.config.max_traces:
                break
            subset = traces[:num_traces]
            diff_traces, key_rankings = self.attack_single_byte(subset, target_byte)
            best_key = np.argmax(key_rankings)
            result = AttackResult(
                recovered_key=np.array([best_key]),
                correlation_traces=diff_traces,
                key_rankings=key_rankings,
                success_rate=key_rankings[best_key] / np.sum(key_rankings),
                num_traces_used=num_traces,
                attack_type="DPA"
            )
            results.append(result)
        return results

class CorrelationPowerAnalysis:
    """Correlation Power Analysis (CPA) implementation."""
    
    def __init__(self, config: AttackConfig):
        self.config = config
    
    def attack_single_byte(self, traces: List[TraceData], target_byte: int) -> Tuple[np.ndarray, np.ndarray]:
        if not traces: raise ValueError("No traces provided")
        waves = np.array([trace.wave for trace in traces])
        plaintexts = np.array([trace.plaintext for trace in traces])
        plaintext_bytes = np.array([
            (pt >> (8 * (15 - target_byte))) & 0xFF for pt in plaintexts
        ])
        num_samples = waves.shape[1]
        correlation_traces = np.zeros((256, num_samples))
        key_rankings = np.zeros(256)
        for key_guess in range(256):
            hyp_power = np.array([
                self.config.power_model.compute(pt_byte, key_guess) 
                for pt_byte in plaintext_bytes
            ])
            for sample_idx in range(num_samples):
                sample_values = waves[:, sample_idx]
                corr = np.corrcoef(sample_values, hyp_power)[0, 1]
                if np.isnan(corr): corr = 0.0
                correlation_traces[key_guess, sample_idx] = corr
            key_rankings[key_guess] = np.max(np.abs(correlation_traces[key_guess]))
        return correlation_traces, key_rankings
    
    def incremental_attack(self, traces: List[TraceData], target_byte: int) -> List[AttackResult]:
        results = []
        increment = self.config.incremental_size
        for num_traces in range(increment, len(traces) + 1, increment):
            if self.config.max_traces and num_traces > self.config.max_traces:
                break
            subset = traces[:num_traces]
            corr_traces, key_rankings = self.attack_single_byte(subset, target_byte)
            best_key = np.argmax(key_rankings)
            result = AttackResult(
                recovered_key=np.array([best_key]),
                correlation_traces=corr_traces,
                key_rankings=key_rankings,
                success_rate=key_rankings[best_key],
                num_traces_used=num_traces,
                attack_type="CPA"
            )
            results.append(result)
        return results

def demonstrate_attacks():
    from sca_framework import TraceCapture
    capture = TraceCapture()
    traces = capture.load_traces(r"C:\Users\Ahmed\Desktop\AdvSecLab\advseceng25-sca-framework-main\src\py\data\traces_1_filtered.npz")
    correct_key = 0x10a5_8869_d74b_e5a3_74cf_867c_fb47_3859
    correct_key_byte = list(correct_key.to_bytes(16, 'big'))[0]
    
    # CPA test
    cpa_config = AttackConfig(power_model=HammingWeightEncModel(), max_traces=2000)
    cpa_attacker = CorrelationPowerAnalysis(cpa_config)
    cpa_results = cpa_attacker.incremental_attack(traces, 0)
    best_cpa = max(cpa_results, key=lambda r: r.success_rate)
    print(f"CPA: Key=0x{best_cpa.recovered_key[0]:02x}, Correlation={best_cpa.success_rate:.4f}")
    
    # DPA test
    dpa_config = AttackConfig(power_model=HammingWeightEncModel(), max_traces=2000)
    dpa_attacker = DifferentialPowerAnalysis(dpa_config)
    dpa_results = dpa_attacker.incremental_attack(traces, 0)
    best_dpa = max(dpa_results, key=lambda r: r.success_rate)
    print(f"DPA: Key=0x{best_dpa.recovered_key[0]:02x}, Ranking={best_dpa.success_rate:.4f}")

if __name__ == "__main__":
    demonstrate_attacks()