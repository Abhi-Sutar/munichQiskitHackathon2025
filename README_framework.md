# Transpiler Performance Comparison Framework

## Overview

This framework provides a comprehensive object-oriented solution for comparing custom quantum circuit transpilation passes with Qiskit's default transpiler. The system is designed to be modular, extensible, and easy to use in Jupyter notebooks for step-by-step analysis.

## File Structure

```
src/
├── test_circuit_generator.py    # Circuit generation utilities
├── custom_pass.py               # Custom transpilation passes
├── performance_metrics.py       # Main benchmarking framework
└── transpiler_comparison_demo.ipynb  # Example usage notebook
```

## Core Components

### 1. TranspilerBenchmark Class

The main class that orchestrates the entire benchmarking process:

```python
from performance_metrics import TranspilerBenchmark

# Create benchmark instance
benchmark = TranspilerBenchmark(backend_name="FakeManila", max_qubits=5)

# Run complete benchmark
results = benchmark.run_full_benchmark()

# Generate summary table
summary_df = benchmark.generate_summary_table()
```

**Key Features:**
- Object-oriented design with clean separation of concerns
- Support for multiple fake backends (FakeManila, FakeTorino, etc.)
- Automated noise model generation
- Comprehensive timing measurements
- Error handling and graceful degradation

### 2. Data Structures

**CircuitMetrics:** Stores performance metrics for individual circuits
```python
@dataclass
class CircuitMetrics:
    depth: int
    num_gates: int
    num_cx: int
    num_single_qubit_gates: int
    transpilation_time: float
    error_message: Optional[str] = None
```

**SimulationResults:** Stores simulation outcomes (extensible for future use)
```python
@dataclass
class SimulationResults:
    execution_time: float
    fidelity: Optional[float] = None
    success_probability: Optional[float] = None
    error_message: Optional[str] = None
```

### 3. Test Circuit Generation

Generates diverse quantum circuits for comprehensive testing:

- **Variational Circuits:** EfficientSU2 with different parameters
- **Quantum Volume Circuits:** With specific seeds for reproducibility
- **Random Circuits:** With controlled randomness for consistent testing

```python
from test_circuit_generator import generate_all_test_circuits

circuits = generate_all_test_circuits(max_qubits_for_backend=5)
```

### 4. Custom Transpilation Pass

Example custom optimization pass that demonstrates:
- CNOT gate cancellation
- Modern Qiskit DAG traversal
- Integration with standard transpilation pipeline

```python
from custom_pass import get_custom_pass_manager, MyOptimizationPass

# Get custom pass manager
custom_pm = get_custom_pass_manager(backend)
optimized_circuit = custom_pm.run(original_circuit)
```

## Usage Examples

### Quick Start

```python
from performance_metrics import create_performance_comparison

# Run complete benchmark with one function call
benchmark = create_performance_comparison(
    backend_name="FakeManila",
    max_qubits=5,
    save_results=True
)
```

### Step-by-Step Analysis

```python
from performance_metrics import TranspilerBenchmark

# 1. Create benchmark
benchmark = TranspilerBenchmark("FakeManila", max_qubits=5)

# 2. Generate or load circuits
circuits = benchmark.generate_test_circuits()

# 3. Run benchmark
results = benchmark.run_full_benchmark(circuits)

# 4. Analyze results
benchmark.print_summary()
summary_df = benchmark.generate_summary_table()

# 5. Save results
benchmark.save_results("my_results.csv")
```

### Custom Circuit Testing

```python
from qiskit import QuantumCircuit

# Create custom test circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 1)  # Redundant gate for optimization testing
qc.name = "my_test_circuit"

# Test single circuit
benchmark = TranspilerBenchmark("FakeManila")
results = benchmark.benchmark_single_circuit(qc)
```

## Results and Outputs

### 1. Console Output
- Real-time progress updates
- Formatted summary tables
- Performance comparisons
- Error reporting

### 2. CSV Export
- Complete metrics for all circuits and methods
- Timing information
- Gate count statistics
- Ready for further analysis in Excel/Python

### 3. DataFrame Integration
- Pandas-compatible data structures
- Easy visualization with matplotlib/seaborn
- Statistical analysis capabilities

## Performance Metrics Tracked

### Circuit-Level Metrics
- **Depth:** Circuit execution depth
- **Total Gates:** Number of quantum gates
- **CX Gates:** Number of two-qubit gates
- **Single-Qubit Gates:** Number of single-qubit gates

### Transpilation Metrics
- **Transpilation Time:** Time taken for transpilation
- **Error Handling:** Graceful handling of transpilation failures
- **Comparison:** Custom vs Qiskit optimization levels 0-3

### Improvement Analysis
- **Depth Improvement:** Reduction in circuit depth
- **Gate Count Reduction:** Optimization in gate count
- **Relative Performance:** Comparison across different methods

## Extensibility

The framework is designed for easy extension:

### Adding New Backends
```python
# In TranspilerBenchmark._setup_backend()
backend_map = {
    "FakeManila": FakeManilaV2,
    "FakeTorino": FakeTorino,
    "YourNewBackend": YourNewBackend,  # Add here
}
```

### Adding New Metrics
```python
# Extend CircuitMetrics dataclass
@dataclass
class CircuitMetrics:
    # ... existing fields ...
    your_new_metric: float = 0.0
```

### Adding Custom Passes
```python
# In custom_pass.py
class YourCustomPass(TransformationPass):
    def run(self, dag):
        # Your optimization logic
        return optimized_dag

# Add to get_custom_pass_manager()
pm.append(YourCustomPass())
```

## Integration with Jupyter Notebooks

The framework is optimized for notebook usage:

1. **Import modules:** Clean imports with no setup required
2. **Step-by-step execution:** Run benchmarks incrementally
3. **Visualization ready:** DataFrames work directly with plotting libraries
4. **Interactive analysis:** Explore results interactively

See `transpiler_comparison_demo.ipynb` for a complete example.

## Performance Characteristics

- **Scalable:** Handles circuits from 2-10+ qubits efficiently
- **Fast:** Optimized for quick iteration during development
- **Memory Efficient:** Minimal memory footprint for large circuit sets
- **Robust:** Comprehensive error handling and recovery

## Best Practices

1. **Start Small:** Begin with max_qubits=3-5 for initial testing
2. **Incremental Testing:** Test single circuits before full benchmarks
3. **Save Results:** Always save results for later analysis
4. **Version Control:** Track changes to custom passes with results
5. **Documentation:** Document custom optimizations and their expected benefits

This framework provides a solid foundation for quantum transpiler research and development, enabling systematic comparison and optimization of quantum circuit compilation strategies.
