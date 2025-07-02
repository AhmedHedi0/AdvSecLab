"""
Side-Channel Attack Algorithms Module
====================================

This module implements Differential Power Analysis (DPA) and Correlation Power Analysis (CPA)
algorithms for recovering secret keys from power traces.

Both algorithms support:
- Flexible number of traces processing
- Incremental processing capabilities
- Multiple power models (Hamming weight, Hamming distance)
- Configurable intermediate result output
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sca_framework import TraceData, PowerModel, HammingWeightModel, HammingDistanceModel, AttackResult
import logging

logger = logging.getLogger(__name__)

@dataclass
class AttackConfig:
    """Configuration for side-channel attacks."""
    power_model: PowerModel
    target_byte: int = 0
    incremental_size: int = 1000
    report_interval: int = 500
    max_traces: Optional[int] = None

class DifferentialPowerAnalysis:
    """
    Differential Power Analysis (DPA) implementation.
    
    DPA works by partitioning traces into two sets based on a selection function
    and computing the difference of means between the sets.
    """
    
    def __init__(self, config: AttackConfig):
        self.config = config
        self.results_history = []
    
    def selection_function(self, plaintext_byte: int, key_guess: int, bit_position: int = 0) -> int:
        """
        Selection function for DPA.
        
        Args:
            plaintext_byte: Input byte value
            key_guess: Hypothetical key byte
            bit_position: Which bit to use for partitioning (0-7)
            
        Returns:
            0 or 1 based on the bit value
        """
        if isinstance(self.config.power_model, HammingWeightModel):
            sbox_output = self.config.power_model.sbox[plaintext_byte ^ key_guess]
            return (sbox_output >> bit_position) & 1
        else:
            # For other models, use the LSB of the power model output
            power_value = int(self.config.power_model.compute(plaintext_byte, key_guess))
            return (power_value >> bit_position) & 1
    
    def attack_single_byte(self, traces: List[TraceData], 
                          target_byte: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform DPA attack on a single key byte.
        
        Args:
            traces: List of power traces
            target_byte: Which byte position to attack (0-15)
            
        Returns:
            Tuple of (differential_traces, key_rankings)
        """
        if target_byte is None:
            target_byte = self.config.target_byte
        
        if not traces:
            raise ValueError("No traces provided")
        
        # Extract data
        waves = np.array([trace.wave for trace in traces])
        plaintexts = np.array([trace.plaintext for trace in traces])
        
        # Extract target byte from plaintexts
        plaintext_bytes = []
        for pt in plaintexts:
            pt_bytes = list(pt.to_bytes(16, 'big'))
            plaintext_bytes.append(pt_bytes[target_byte])
        plaintext_bytes = np.array(plaintext_bytes)
        
        num_samples = waves.shape[1]
        num_traces = len(traces)
        
        # Results storage
        differential_traces = np.zeros((256, num_samples))
        key_rankings = np.zeros(256)
        
        logger.info(f"Starting DPA attack on byte {target_byte} with {num_traces} traces")
        
        # Test each key hypothesis
        for key_guess in range(256):
            # Partition traces based on selection function
            set_0_indices = []
            set_1_indices = []
            
            for i, pt_byte in enumerate(plaintext_bytes):
                selection_bit = self.selection_function(pt_byte, key_guess, bit_position=0)
                if selection_bit == 0:
                    set_0_indices.append(i)
                else:
                    set_1_indices.append(i)
            
            if len(set_0_indices) > 0 and len(set_1_indices) > 0:
                # Compute differential trace
                mean_0 = np.mean(waves[set_0_indices], axis=0)
                mean_1 = np.mean(waves[set_1_indices], axis=0)
                differential_traces[key_guess] = mean_1 - mean_0
                
                # Compute ranking metric (max absolute difference)
                key_rankings[key_guess] = np.max(np.abs(differential_traces[key_guess]))
            else:
                logger.warning(f"Unbalanced partition for key guess {key_guess}")
                key_rankings[key_guess] = 0
        
        logger.info(f"DPA attack completed. Best key guess: {np.argmax(key_rankings)}")
        return differential_traces, key_rankings
    
    def incremental_attack(self, traces: List[TraceData], 
                          target_byte: int = None) -> List[AttackResult]:
        """
        Perform incremental DPA attack with configurable intervals.
        
        Args:
            traces: List of power traces
            target_byte: Which byte position to attack
            
        Returns:
            List of attack results at each interval
        """
        if target_byte is None:
            target_byte = self.config.target_byte
        
        results = []
        increment = self.config.incremental_size
        
        for num_traces in range(increment, len(traces) + 1, increment):
            if self.config.max_traces and num_traces > self.config.max_traces:
                break
            
            # Use subset of traces
            subset_traces = traces[:num_traces]
            
            # Perform attack
            diff_traces, key_rankings = self.attack_single_byte(subset_traces, target_byte)
            
            # Find best key
            best_key = np.argmax(key_rankings)
            success_rate = key_rankings[best_key] / np.sum(key_rankings) if np.sum(key_rankings) > 0 else 0
            
            # Create result
            result = AttackResult(
                recovered_key=np.array([best_key]),
                correlation_traces=diff_traces,
                key_rankings=key_rankings,
                success_rate=success_rate,
                num_traces_used=num_traces,
                attack_type="DPA"
            )
            
            results.append(result)
            
            # Report progress
            if num_traces % self.config.report_interval == 0:
                logger.info(f"DPA progress: {num_traces} traces, best key: {best_key}, "
                           f"ranking: {key_rankings[best_key]:.4f}")
        
        self.results_history.extend(results)
        return results

class CorrelationPowerAnalysis:
    """
    Correlation Power Analysis (CPA) implementation.
    
    CPA computes the correlation coefficient between power traces and
    hypothetical power consumption based on a power model.
    """
    
    def __init__(self, config: AttackConfig):
        self.config = config
        self.results_history = []
    
    def attack_single_byte(self, traces: List[TraceData], 
                          target_byte: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform CPA attack on a single key byte.
        
        Args:
            traces: List of power traces
            target_byte: Which byte position to attack (0-15)
            
        Returns:
            Tuple of (correlation_traces, key_rankings)
        """
        if target_byte is None:
            target_byte = self.config.target_byte
        
        if not traces:
            raise ValueError("No traces provided")
        
        # Extract data
        waves = np.array([trace.wave for trace in traces])
        plaintexts = np.array([trace.plaintext for trace in traces])
        
        # Extract target byte from plaintexts
        plaintext_bytes = []
        for pt in plaintexts:
            pt_bytes = list(pt.to_bytes(16, 'big'))
            plaintext_bytes.append(pt_bytes[target_byte])
        plaintext_bytes = np.array(plaintext_bytes)
        
        num_samples = waves.shape[1]
        num_traces = len(traces)
        
        # Results storage
        correlation_traces = np.zeros((256, num_samples))
        key_rankings = np.zeros(256)
        
        logger.info(f"Starting CPA attack on byte {target_byte} with {num_traces} traces")
        
        # Test each key hypothesis
        for key_guess in range(256):
            # Compute hypothetical power consumption
            hyp_power = np.array([
                self.config.power_model.compute(pt_byte, key_guess) 
                for pt_byte in plaintext_bytes
            ])
            
            # Compute correlation for each sample point
            for sample_idx in range(num_samples):
                sample_values = waves[:, sample_idx]
                
                # Calculate Pearson correlation coefficient
                correlation = np.corrcoef(sample_values, hyp_power)[0, 1]
                
                # Handle NaN values (can occur with constant signals)
                if np.isnan(correlation):
                    correlation = 0.0
                
                correlation_traces[key_guess, sample_idx] = correlation
            
            # Ranking metric: maximum absolute correlation
            key_rankings[key_guess] = np.max(np.abs(correlation_traces[key_guess]))
        
        logger.info(f"CPA attack completed. Best key guess: {np.argmax(key_rankings)}")
        return correlation_traces, key_rankings
    
    def incremental_attack(self, traces: List[TraceData], 
                          target_byte: int = None) -> List[AttackResult]:
        """
        Perform incremental CPA attack with configurable intervals.
        
        Args:
            traces: List of power traces
            target_byte: Which byte position to attack
            
        Returns:
            List of attack results at each interval
        """
        if target_byte is None:
            target_byte = self.config.target_byte
        
        results = []
        increment = self.config.incremental_size
        
        for num_traces in range(increment, len(traces) + 1, increment):
            if self.config.max_traces and num_traces > self.config.max_traces:
                break
            
            # Use subset of traces
            subset_traces = traces[:num_traces]
            
            # Perform attack
            corr_traces, key_rankings = self.attack_single_byte(subset_traces, target_byte)
            
            # Find best key
            best_key = np.argmax(key_rankings)
            success_rate = key_rankings[best_key]  # Correlation coefficient as success metric
            
            # Create result
            result = AttackResult(
                recovered_key=np.array([best_key]),
                correlation_traces=corr_traces,
                key_rankings=key_rankings,
                success_rate=success_rate,
                num_traces_used=num_traces,
                attack_type="CPA"
            )
            
            results.append(result)
            
            # Report progress
            if num_traces % self.config.report_interval == 0:
                logger.info(f"CPA progress: {num_traces} traces, best key: {best_key}, "
                           f"correlation: {key_rankings[best_key]:.4f}")
        
        self.results_history.extend(results)
        return results

class FullKeyRecovery:
    """
    Full 16-byte AES key recovery using DPA or CPA.
    """
    
    def __init__(self, attack_type: str = "CPA", power_model: PowerModel = None):
        self.attack_type = attack_type.upper()
        self.power_model = power_model or HammingWeightModel()
        self.recovered_key = np.zeros(16, dtype=np.uint8)
        self.byte_results = {}
    
    def recover_full_key(self, traces: List[TraceData], 
                        incremental_size: int = 1000) -> Dict[int, List[AttackResult]]:
        """
        Recover all 16 bytes of the AES key.
        
        Args:
            traces: List of power traces
            incremental_size: Number of traces to process incrementally
            
        Returns:
            Dictionary mapping byte position to attack results
        """
        all_results = {}
        
        for byte_pos in range(16):
            logger.info(f"Attacking key byte {byte_pos}")
            
            # Configure attack
            config = AttackConfig(
                power_model=self.power_model,
                target_byte=byte_pos,
                incremental_size=incremental_size,
                report_interval=incremental_size,
                max_traces=len(traces)
            )
            
            # Choose attack algorithm
            if self.attack_type == "DPA":
                attacker = DifferentialPowerAnalysis(config)
            else:  # CPA
                attacker = CorrelationPowerAnalysis(config)
            
            # Perform incremental attack
            results = attacker.incremental_attack(traces, byte_pos)
            all_results[byte_pos] = results
            
            # Store best result for this byte
            if results:
                best_result = max(results, key=lambda r: r.success_rate)
                self.recovered_key[byte_pos] = best_result.recovered_key[0]
                self.byte_results[byte_pos] = best_result
        
        logger.info(f"Full key recovery completed using {self.attack_type}")
        logger.info(f"Recovered key: {self.recovered_key.hex()}")
        
        return all_results

def demonstrate_attacks():
    """Demonstrate DPA and CPA attack implementations."""
    print("Side-Channel Attack Algorithms Demonstration")
    print("=" * 45)
    
    # Load traces
    from sca_framework import TraceCapture
    capture = TraceCapture()
    
    try:
        traces = capture.load_traces("/home/ubuntu/sca_framework/demo_traces.npz")
        print(f"Loaded {len(traces)} traces for attack demonstration")
    except Exception as e:
        print(f"Error loading traces: {e}")
        return
    
    # Known key from simulation
    correct_key = 0x2b7e151628aed2a6abf7158809cf4f3c
    correct_key_bytes = list(correct_key.to_bytes(16, 'big'))
    
    print(f"Target key byte 0: 0x{correct_key_bytes[0]:02x}")
    
    # Test CPA attack
    print("\nTesting CPA Attack...")
    cpa_config = AttackConfig(
        power_model=HammingWeightModel(),
        target_byte=0,
        incremental_size=100,
        report_interval=100,
        max_traces=500
    )
    
    cpa_attacker = CorrelationPowerAnalysis(cpa_config)
    cpa_results = cpa_attacker.incremental_attack(traces, 0)
    
    if cpa_results:
        best_cpa = max(cpa_results, key=lambda r: r.success_rate)
        print(f"CPA Best result: Key=0x{best_cpa.recovered_key[0]:02x}, "
              f"Correlation={best_cpa.success_rate:.4f}, "
              f"Traces={best_cpa.num_traces_used}")
        print(f"CPA Success: {'YES' if best_cpa.recovered_key[0] == correct_key_bytes[0] else 'NO'}")
    
    # Test DPA attack
    print("\nTesting DPA Attack...")
    dpa_config = AttackConfig(
        power_model=HammingWeightModel(),
        target_byte=0,
        incremental_size=100,
        report_interval=100,
        max_traces=500
    )
    
    dpa_attacker = DifferentialPowerAnalysis(dpa_config)
    dpa_results = dpa_attacker.incremental_attack(traces, 0)
    
    if dpa_results:
        best_dpa = max(dpa_results, key=lambda r: r.success_rate)
        print(f"DPA Best result: Key=0x{best_dpa.recovered_key[0]:02x}, "
              f"Ranking={best_dpa.success_rate:.4f}, "
              f"Traces={best_dpa.num_traces_used}")
        print(f"DPA Success: {'YES' if best_dpa.recovered_key[0] == correct_key_bytes[0] else 'NO'}")
    
    print("\nAttack algorithms demonstration completed!")
    return cpa_results, dpa_results

if __name__ == "__main__":
    demonstrate_attacks()

