"""
Performance Metrics Module for Quantum Circuit Transpiler Comparison

This module provides a comprehensive framework for comparing custom transpilation
passes with Qiskit's default transpiler across multiple optimization levels.
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager, TranspilerError
from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeTorino

# Import our custom modules
from test_circuit_generator import generate_all_test_circuits
from custom_pass import get_custom_pass_manager


@dataclass
class CircuitMetrics:
    """Data structure to store circuit performance metrics."""
    depth: int
    num_qbits: int
    num_gates: int
    num_cx: int
    num_single_qubit_gates: int
    transpilation_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return asdict(self)


@dataclass
class SimulationResults:
    """Data structure to store simulation results."""
    execution_time: float
    fidelity: Optional[float] = None
    success_probability: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return asdict(self)


class TranspilerBenchmark:
    """
    Main class for benchmarking transpiler performance.
    
    This class handles circuit generation, transpilation comparison,
    simulation, and results analysis.
    """
    
    def __init__(self, backend_name: str = "FakeTorino", max_qubits: int = 5):
        """
        Initialize the benchmark with a specific backend.
        
        Args:
            backend_name: Name of the fake backend to use
            max_qubits: Maximum number of qubits for test circuits
        """
        self.backend_name = backend_name
        self.max_qubits = max_qubits
        self.backend = self._setup_backend(backend_name)
        self.noise_model = self._setup_noise_model()
        self.optimization_levels = [0, 1, 2, 3]
        self.results = {}
        
    def _setup_backend(self, backend_name: str) -> Backend:
        """Setup the quantum backend for testing."""
        backend_map = {
            "FakeManila": FakeManilaV2,
            "FakeManilaV2": FakeManilaV2,
            "FakeTorino": FakeTorino,
        }
        
        if backend_name not in backend_map:
            raise ValueError(f"Backend {backend_name} not supported. "
                           f"Available: {list(backend_map.keys())}")
        
        return backend_map[backend_name]()
    
    def _setup_noise_model(self) -> NoiseModel:
        """Setup noise model from the backend."""
        try:
            return NoiseModel.from_backend(self.backend)
        except Exception:
            # Fallback to a simple custom noise model
            return self._create_simple_noise_model()
    
    def _create_simple_noise_model(self) -> NoiseModel:
        """Create a simple custom noise model for demonstration."""
        noise_model = NoiseModel()
        
        # Add depolarizing error to single-qubit gates
        error_1q = depolarizing_error(0.001, 1)  # 0.1% error
        noise_model.add_all_qubit_quantum_error(error_1q, ['rz', 'sx', 'x'])
        
        # Add depolarizing error to two-qubit gates
        error_2q = depolarizing_error(0.01, 2)  # 1% error
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        
        # Add readout error
        prob_meas0given1 = 0.05
        prob_meas1given0 = 0.01
        readout_error = ReadoutError([[1 - prob_meas1given0, prob_meas1given0],
                                     [prob_meas0given1, 1 - prob_meas0given1]])
        noise_model.add_all_qubit_readout_error(readout_error)
        
        return noise_model
    
    def generate_test_circuits(self) -> List[QuantumCircuit]:
        """Generate test circuits for benchmarking."""
        return generate_all_test_circuits(max_qubits_for_backend=self.max_qubits)
    
    def _get_circuit_metrics(self, circuit: QuantumCircuit) -> CircuitMetrics:
        """Extract metrics from a quantum circuit."""
        gate_counts = circuit.count_ops()
        num_single_qubit = sum(count for gate, count in gate_counts.items() 
                              if gate not in ['cx', 'measure', 'barrier'])
        
        return CircuitMetrics(
            depth=circuit.depth(),
            num_gates=circuit.size(),
            num_cx=gate_counts.get('cx', 0),
            num_single_qubit_gates=num_single_qubit,
            transpilation_time=0.0  # Will be set during transpilation
        )
    
    def _transpile_with_timing(self, circuit: QuantumCircuit, 
                             pass_manager: PassManager) -> tuple[QuantumCircuit, float]:
        """Transpile circuit and measure timing."""
        start_time = time.time()
        transpiled_circuit = pass_manager.run(circuit)
        transpilation_time = time.time() - start_time
        return transpiled_circuit, transpilation_time
    
    def _transpile_with_qiskit(self, circuit: QuantumCircuit, 
                             optimization_level: int) -> tuple[QuantumCircuit, float]:
        """Transpile with Qiskit's default transpiler and measure timing."""
        start_time = time.time()
        transpiled_circuit = transpile(circuit, self.backend, 
                                     optimization_level=optimization_level)
        transpilation_time = time.time() - start_time
        return transpiled_circuit, transpilation_time
    
    def benchmark_single_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Benchmark a single circuit against all transpilation methods.
        
        Args:
            circuit: The quantum circuit to benchmark
            
        Returns:
            Dictionary containing all metrics for this circuit
        """
        print(f"\n--- Processing Circuit: {circuit.name} ---")
        
        circuit_results = {
            'circuit_name': circuit.name,
            'original': self._get_circuit_metrics(circuit).to_dict()
        }
        
        # Test custom transpiler
        try:
            custom_pm = get_custom_pass_manager(self.backend)
            transpiled_custom, custom_time = self._transpile_with_timing(circuit, custom_pm)
            
            custom_metrics = self._get_circuit_metrics(transpiled_custom)
            custom_metrics.transpilation_time = custom_time
            
            circuit_results['custom'] = custom_metrics.to_dict()
            circuit_results['custom']['transpiled_circuit'] = transpiled_custom
            
            print(f"Custom Transpiler: Depth={custom_metrics.depth}, "
                  f"Gates={custom_metrics.num_gates}, CX={custom_metrics.num_cx}, "
                  f"Time={custom_time:.4f}s")
                  
        except TranspilerError as e:
            print(f"Custom transpiler failed: {e}")
            circuit_results['custom'] = CircuitMetrics(
                depth=0, num_gates=0, num_cx=0, num_single_qubit_gates=0,
                transpilation_time=0.0, error_message=str(e)
            ).to_dict()
        
        # Test Qiskit's default transpiler at different optimization levels
        circuit_results['qiskit'] = {}
        for level in self.optimization_levels:
            try:
                transpiled_qiskit, qiskit_time = self._transpile_with_qiskit(circuit, level)
                
                qiskit_metrics = self._get_circuit_metrics(transpiled_qiskit)
                qiskit_metrics.transpilation_time = qiskit_time
                
                circuit_results['qiskit'][f'level_{level}'] = qiskit_metrics.to_dict()
                circuit_results['qiskit'][f'level_{level}']['transpiled_circuit'] = transpiled_qiskit
                
                print(f"Qiskit (Level {level}): Depth={qiskit_metrics.depth}, "
                      f"Gates={qiskit_metrics.num_gates}, CX={qiskit_metrics.num_cx}, "
                      f"Time={qiskit_time:.4f}s")
                      
            except TranspilerError as e:
                print(f"Qiskit transpiler (level {level}) failed: {e}")
                circuit_results['qiskit'][f'level_{level}'] = CircuitMetrics(
                    depth=0, num_gates=0, num_cx=0, num_single_qubit_gates=0,
                    transpilation_time=0.0, error_message=str(e)
                ).to_dict()
        
        return circuit_results
    
    def run_full_benchmark(self, circuits: Optional[List[QuantumCircuit]] = None) -> Dict[str, Any]:
        """
        Run the complete benchmark on all test circuits.
        
        Args:
            circuits: Optional list of circuits to test. If None, generates test circuits.
            
        Returns:
            Complete results dictionary
        """
        if circuits is None:
            circuits = self.generate_test_circuits()
        
        print(f"Starting benchmark with {len(circuits)} circuits on {self.backend_name}")
        print(f"Backend coupling map: {self.backend.coupling_map}")
        print(f"Backend basis gates: {self.backend.basis_gates}")
        
        for circuit in circuits:
            circuit_results = self.benchmark_single_circuit(circuit)
            self.results[circuit.name] = circuit_results
        
        return self.results
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of all results.
        
        Returns:
            pandas DataFrame with summary metrics
        """
        summary_data = []
        
        for circuit_name, data in self.results.items():
            row = {
                'Circuit': circuit_name,
                'Original_Depth': data['original']['depth'],
                'Original_Gates': data['original']['num_gates'],
                'Original_CX': data['original']['num_cx'],
            }
            
            # Custom transpiler results
            if 'custom' in data and not data['custom'].get('error_message'):
                row.update({
                    'Custom_Depth': data['custom']['depth'],
                    'Custom_Gates': data['custom']['num_gates'],
                    'Custom_CX': data['custom']['num_cx'],
                    'Custom_Time': data['custom']['transpilation_time'],
                })
            else:
                row.update({
                    'Custom_Depth': 'Error',
                    'Custom_Gates': 'Error',
                    'Custom_CX': 'Error',
                    'Custom_Time': 'Error',
                })
            
            # Qiskit transpiler results for each level
            for level in self.optimization_levels:
                level_key = f'level_{level}'
                if (level_key in data['qiskit'] and 
                    not data['qiskit'][level_key].get('error_message')):
                    row.update({
                        f'Qiskit_L{level}_Depth': data['qiskit'][level_key]['depth'],
                        f'Qiskit_L{level}_Gates': data['qiskit'][level_key]['num_gates'],
                        f'Qiskit_L{level}_CX': data['qiskit'][level_key]['num_cx'],
                        f'Qiskit_L{level}_Time': data['qiskit'][level_key]['transpilation_time'],
                    })
                else:
                    row.update({
                        f'Qiskit_L{level}_Depth': 'Error',
                        f'Qiskit_L{level}_Gates': 'Error',
                        f'Qiskit_L{level}_CX': 'Error',
                        f'Qiskit_L{level}_Time': 'Error',
                    })
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, filepath: str = "transpiler_benchmark_results.csv"):
        """
        Save the summary results to a CSV file.
        
        Args:
            filepath: Path where to save the results
        """
        summary_df = self.generate_summary_table()
        summary_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print a formatted summary of the benchmark results."""
        print("\n" + "="*80)
        print(" TRANSPILER BENCHMARK SUMMARY")
        print("="*80)
        
        for circuit_name, data in self.results.items():
            print(f"\nCircuit: {circuit_name}")
            print("-" * 50)
            
            # Original metrics
            orig = data['original']
            print(f"Original:    Depth={orig['depth']:3d}, Gates={orig['num_gates']:3d}, CX={orig['num_cx']:3d}")
            
            # Custom transpiler
            if 'custom' in data and not data['custom'].get('error_message'):
                custom = data['custom']
                depth_improvement = orig['depth'] - custom['depth']
                gates_improvement = orig['num_gates'] - custom['num_gates']
                print(f"Custom:      Depth={custom['depth']:3d} ({depth_improvement:+d}), "
                      f"Gates={custom['num_gates']:3d} ({gates_improvement:+d}), "
                      f"CX={custom['num_cx']:3d}, Time={custom['transpilation_time']:.4f}s")
            else:
                print("Custom:      FAILED")
            
            # Qiskit levels
            for level in self.optimization_levels:
                level_key = f'level_{level}'
                if (level_key in data['qiskit'] and 
                    not data['qiskit'][level_key].get('error_message')):
                    qiskit = data['qiskit'][level_key]
                    depth_improvement = orig['depth'] - qiskit['depth']
                    gates_improvement = orig['num_gates'] - qiskit['num_gates']
                    print(f"Qiskit L{level}:   Depth={qiskit['depth']:3d} ({depth_improvement:+d}), "
                          f"Gates={qiskit['num_gates']:3d} ({gates_improvement:+d}), "
                          f"CX={qiskit['num_cx']:3d}, Time={qiskit['transpilation_time']:.4f}s")
                else:
                    print(f"Qiskit L{level}:   FAILED")


def create_performance_comparison(backend_name: str = "FakeTorino", 
                                max_qubits: int = 5,
                                save_results: bool = True) -> TranspilerBenchmark:
    """
    Convenience function to create and run a complete performance comparison.
    
    Args:
        backend_name: Name of the fake backend to use
        max_qubits: Maximum number of qubits for test circuits
        save_results: Whether to save results to CSV
        
    Returns:
        TranspilerBenchmark instance with completed results
    """
    benchmark = TranspilerBenchmark(backend_name=backend_name, max_qubits=max_qubits)
    benchmark.run_full_benchmark()
    benchmark.print_summary()
    
    if save_results:
        benchmark.save_results(f"transpiler_comparison_{backend_name}_results.csv")
    
    return benchmark


# Example usage and testing
if __name__ == "__main__":
    # Create and run benchmark
    benchmark = create_performance_comparison(
        backend_name="FakeTorino",
        max_qubits=5,
        save_results=True
    )
    
    # Display summary table
    summary_df = benchmark.generate_summary_table()
    print("\n" + "="*80)
    print(" SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
