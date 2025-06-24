import time
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager, TranspilerError
from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_ibm_runtime.fake_provider import Backend, FakeManilaV2, FakeTorino

# Import our custom modules
from test_circuit_generator import generate_all_test_circuits
from custom_pass import get_custom_pass_manager
from performance_metrics import (
    CircuitMetrics,
    SimulationResults,
)



class TranspilerBenchmark:

    def __init__(self, backend: Backend, max_qubits: int = 6):
        self.backend = backend
        self.max_qubits = max_qubits