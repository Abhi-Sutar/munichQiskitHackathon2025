import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    QFT,  # Standard Algorithm: Quantum Fourier Transform
    efficient_su2,  # Variational Circuit (hardware-efficient ansatz) - function
    QuantumVolume,  # Benchmarking Circuit: Quantum Volume model circuit
)

# Set NumPy seed at the very beginning of this file for reproducibility.
# This ensures that any random elements in circuit generation (e.g., in get_random_circuit
# or within QuantumVolume's internal randomness) are consistent across runs.
np.random.seed(42)


def get_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Generates a Quantum Fourier Transform (QFT) circuit.

    Args:
        num_qubits (int): The number of qubits for the QFT circuit.

    Returns:
        QuantumCircuit: The QFT circuit.
    """
    qc = QFT(num_qubits).decompose()  # Decompose to elementary gates
    qc.name = f"QFT_{num_qubits}Q"
    return qc


def get_efficient_su2_circuit(
    num_qubits: int, reps: int = 1, entanglement: str = "linear"
) -> QuantumCircuit:
    """
    Generates an EfficientSU2 variational circuit (hardware-efficient ansatz).

    Args:
        num_qubits (int): The number of qubits for the circuit.
        reps (int): The number of repetitions (layers) of the rotation and entanglement blocks.
        entanglement (str): The entanglement pattern ('linear', 'circular', 'full', etc.).

    Returns:
        QuantumCircuit: The EfficientSU2 circuit.
    """
    qc = efficient_su2(num_qubits, reps=reps, entanglement=entanglement).decompose()
    qc.name = f"EfficientSU2_{num_qubits}Q_R{reps}_{entanglement}"
    return qc


def get_bell_state_circuit() -> QuantumCircuit:
    """
    Generates a 2-qubit Bell state (Phi+) preparation circuit.
    Creates the state |00⟩ + |11⟩ using H and CNOT gates.

    Returns:
        QuantumCircuit: The Bell state circuit.
    """
    qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
    qc.h(0)  # Put first qubit in superposition
    qc.cx(0, 1)  # Entangle with second qubit
    qc.name = "BellState"
    return qc


def get_ghz_state_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Generates a Greenberger–Horne–Zeilinger (GHZ) state preparation circuit.
    Creates the state |00...0⟩ + |11...1⟩ using H and CX gates.

    Args:
        num_qubits (int): The number of qubits for the GHZ state.

    Returns:
        QuantumCircuit: The GHZ state circuit.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)  # Include classical bits
    qc.h(0)  # Put first qubit in superposition
    # Entangle all other qubits with the first one
    for i in range(1, num_qubits):
        qc.cx(0, i)
    qc.name = f"GHZState_{num_qubits}Q"
    return qc


def get_qv_circuit(
    num_qubits: int, depth: int = None, seed: int = None
) -> QuantumCircuit:
    """
    Generates a Quantum Volume (QV) model circuit.
    The internal randomness for gate selection within QV is also seeded by 'seed'.

    Args:
        num_qubits (int): The number of qubits for the QV circuit.
        depth (int, optional): The depth of the QV circuit. If None, it's chosen automatically.
        seed (int, optional): A seed for the random number generator used in circuit generation.

    Returns:
        QuantumCircuit: The Quantum Volume circuit.
    """
    # QuantumVolume constructor handles its own internal randomness if seed is provided
    qc = QuantumVolume(num_qubits, depth=depth, seed=seed)
    qc.name = f"QV_{num_qubits}Q_D{depth or 'auto'}_S{seed}"
    return qc


def get_random_circuit(num_qubits: int, depth: int, seed: int = None) -> QuantumCircuit:
    """
    Generates a random quantum circuit with RX, RZ, and CX gates.

    Args:
        num_qubits (int): The number of qubits for the circuit.
        depth (int): The number of layers (depth) of random gates.
        seed (int, optional): A seed for the random number generator.

    Returns:
        QuantumCircuit: The random quantum circuit.
    """
    qc = QuantumCircuit(
        num_qubits, num_qubits
    )  # Include classical bits for measurement
    # Use a local random number generator for this specific function for isolation
    # and to ensure reproducibility even if np.random.seed is not globally set.
    rng = np.random.default_rng(seed)

    for _ in range(depth):
        # Apply random single-qubit rotations
        for q in range(num_qubits):
            qc.rx(rng.random() * 2 * np.pi, q)
            qc.rz(rng.random() * 2 * np.pi, q)

        # Apply random two-qubit CX gates
        if num_qubits > 1:
            # Randomly choose two distinct qubits for a CX gate
            q1, q2 = rng.choice(num_qubits, 2, replace=False)
            qc.cx(q1, q2)

    qc.measure_all()  # Add measurements to get classical outcomes
    qc.name = f"Random_{num_qubits}Q_D{depth}_S{seed}"
    return qc


def generate_all_test_circuits(
    max_qubits_for_backend: int = None,
) -> list[QuantumCircuit]:
    """
    Generates a predefined list of test circuits, limiting qubit count if specified.
    This function focuses on circuits that have randomness or variational parameters
    that make them worth saving. Deterministic circuits like QFT, Bell states, and GHZ
    states can be generated on-demand at runtime and don't need to be pre-saved.

    Args:
        max_qubits_for_backend (int, optional): Maximum number of qubits a circuit
            can have to be included. Useful for filtering circuits to a specific backend size.

    Returns:
        list[QuantumCircuit]: A list of generated QuantumCircuit objects.
    """
    circuits = []

    # --- Variational Circuits (worth saving due to parameter configurations) ---
    # EfficientSU2: vary reps and entanglement, limited by max_qubits
    for n_q in range(2, (max_qubits_for_backend or 6) + 1):
        circuits.append(get_efficient_su2_circuit(n_q, reps=1, entanglement="linear"))
        if n_q >= 4:  # Only for slightly larger circuits, try more complex entanglement
            circuits.append(get_efficient_su2_circuit(n_q, reps=2, entanglement="full"))

    # --- Quantum Volume Circuits (truly random, worth saving) ---
    # These use seeds and are considered 'persistently random'
    qv_configs = [
        (4, 4, 10),  # num_qubits, depth, seed
        (5, 5, 20),
        (6, 6, 30),
        (7, 7, 40),
    ]
    for n_q, depth, seed in qv_configs:
        if max_qubits_for_backend is None or n_q <= max_qubits_for_backend:
            circuits.append(get_qv_circuit(n_q, depth=depth, seed=seed))

    # --- Random Circuits (truly random, worth saving) ---
    random_configs = [
        (3, 5, 100),  # num_qubits, depth, seed
        (4, 8, 101),
        (5, 10, 102),
        (6, 12, 103),
        (7, 15, 104),
    ]
    for n_q, depth, seed in random_configs:
        if max_qubits_for_backend is None or n_q <= max_qubits_for_backend:
            # Limit depth for larger random circuits if needed for performance
            effective_depth = min(
                depth, 100
            )  # Arbitrary limit for very deep random circuits on simulators
            circuits.append(get_random_circuit(n_q, depth=effective_depth, seed=seed))

    # Filter out any circuits that might have been generated but exceed max_qubits_for_backend
    # (this is a redundant check if filtering is done during generation, but safer)
    if max_qubits_for_backend is not None:
        circuits = [qc for qc in circuits if qc.num_qubits <= max_qubits_for_backend]

    return circuits


if __name__ == "__main__":
    # Example of how to use this generator:
    print("--- Generating a small set of test circuits (for demonstration) ---")

    # Generate circuits for a hypothetical 5-qubit backend
    test_circs = generate_all_test_circuits(max_qubits_for_backend=5)

    for i, qc in enumerate(test_circs):
        print(
            f"  Circuit {i + 1}: {qc.name} ({qc.num_qubits} qubits, Depth: {qc.depth()})"
        )
        # You can also draw them if you have matplotlib installed
        # qc.draw(output='mpl', filename=f'{qc.name}.png', idle_wires=False)
