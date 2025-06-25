from qiskit.providers import Backend
from qiskit_ibm_runtime.fake_provider import Backend

# Import our custom modules


class TranspilerBenchmark:
    def __init__(self, backend: Backend, max_qubits: int = 6):
        self.backend = backend
        self.max_qubits = max_qubits
