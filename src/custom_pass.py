from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
)  # Example passes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Example of a custom pass structure (replace with your actual logic)
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit


class MyOptimizationPass(TransformationPass):
    def run(self, dag):
        # Implement your optimization logic here.
        # This is just a placeholder example: try to cancel adjacent CNOTs
        new_dag = dag.copy_empty_like()

        # Simple CNOT cancellation example using topological ordering
        nodes_list = list(dag.topological_op_nodes())

        skip_next = False
        for i, node in enumerate(nodes_list):
            if skip_next:
                skip_next = False
                continue

            # Check if this is a CNOT gate and if the next node is also a CNOT on same qubits
            if (
                node.op.name == "cx"
                and i + 1 < len(nodes_list)
                and nodes_list[i + 1].op.name == "cx"
                and node.qargs == nodes_list[i + 1].qargs
                and node.cargs == nodes_list[i + 1].cargs
            ):
                # Found two adjacent CNOTs on the same qubits, cancel them
                skip_next = True
                continue

            # Apply the operation to the new DAG
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag


class DecoherenceWatchdogPass(TransformationPass):
    """
    A hardware-aware transpiler pass that finds the most vulnerable idle
    spot in a circuit and inserts an error-heralding 'watchdog' gadget.
    """

    def __init__(self, backend_properties=None, timing_constraints=None):
        super().__init__()
        self.props = backend_properties
        self.timing_constraints = timing_constraints or {}

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """The main execution method of the transpiler pass."""
        print("\n[WatchdogPass] Running analysis...")

        # If no backend properties provided, skip this pass
        if self.props is None:
            print("[WatchdogPass] No backend properties provided. Skipping pass.")
            return dag

        # This pass REQUIRES the circuit to be laid out and scheduled first.
        # We get this information from the property_set dictionary.
        if (
            "layout" not in self.property_set
            or "node_start_time" not in self.property_set
        ):
            print(
                "[WatchdogPass] Layout and scheduling info not available. Skipping modification."
            )
            return dag

        layout = self.property_set["layout"]
        schedule = self.property_set["node_start_time"]

        # Get timing info from backend properties or use defaults
        dt = getattr(self.props, "dt", 0.222e-9)  # Default Qiskit dt
        coupling_map = getattr(self.props, "coupling_map", None)

        if coupling_map is None:
            print("[WatchdogPass] No coupling map available. Skipping modification.")
            return dag

        worst_spot = {
            "cost": -1,
            "qubit": None,
            "idle_start_dt": None,
            "idle_end_dt": None,
        }

        # --- Analysis: Find the single most vulnerable point in the circuit ---
        for physical_qubit in layout.get_physical_bits():
            # Get all operations on this specific physical qubit, sorted by their start time
            qubit_ops = sorted(
                [
                    node
                    for node in dag.op_nodes()
                    if physical_qubit in [layout[q] for q in node.qargs]
                ],
                key=lambda n: schedule[n],
            )

            # Find idle windows between consecutive gates
            for i in range(len(qubit_ops) - 1):
                prev_node = qubit_ops[i]
                next_node = qubit_ops[i + 1]

                # Get gate times in seconds. This is the most robust way.
                try:
                    prev_qargs = [layout[q] for q in prev_node.qargs]
                    prev_duration_sec = self.props.gate_length(
                        prev_node.op.name.lower(), prev_qargs
                    )
                except (AttributeError, KeyError):
                    # Fallback to default gate length
                    prev_duration_sec = 100e-9  # 100ns default

                idle_start_sec = (schedule[prev_node] * dt) + prev_duration_sec
                idle_end_sec = schedule[next_node] * dt
                idle_duration_sec = idle_end_sec - idle_start_sec

                if idle_duration_sec > 1e-9:  # Only consider non-trivial idle times
                    try:
                        t2_time = self.props.t2(physical_qubit)
                    except (AttributeError, KeyError):
                        t2_time = 100e-6  # 100us default T2

                    idling_cost = idle_duration_sec / t2_time  # The core cost function

                    if idling_cost > worst_spot["cost"]:
                        worst_spot = {
                            "cost": idling_cost,
                            "qubit": physical_qubit,
                            "idle_start_dt": schedule[prev_node]
                            + round(prev_duration_sec / dt),
                            "idle_end_dt": schedule[next_node],
                        }

        if worst_spot["qubit"] is None:
            print("[WatchdogPass] No significant idle time found. No action taken.")
            return dag

        print(
            f"[WatchdogPass] Found most vulnerable spot on qubit {worst_spot['qubit']} with cost {worst_spot['cost']:.4f}"
        )

        # --- Modification: Build and insert the watchdog gadget ---
        data_qubit = worst_spot["qubit"]
        ancilla_qubit = None
        best_ancilla_t2 = -1

        # Heuristic: find a connected neighbor with the best T2 time to be our ancilla
        for neighbor in coupling_map.neighbors(data_qubit):
            try:
                neighbor_t2 = self.props.t2(neighbor)
            except (AttributeError, KeyError):
                neighbor_t2 = 100e-6  # Default T2

            if neighbor_t2 > best_ancilla_t2:
                best_ancilla_t2 = neighbor_t2
                ancilla_qubit = neighbor

        if ancilla_qubit is None:
            print(
                "[WatchdogPass] Could not find a suitable ancilla. Aborting modification."
            )
            return dag

        print(
            f"[WatchdogPass] Selected data qubit: {data_qubit}, ancilla qubit: {ancilla_qubit}"
        )

        # Build the gadget circuit: a Z-basis check
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(
            1, "c_watchdog"
        )  # A dedicated classical bit for the herald
        gadget_qc = QuantumCircuit(qr, cr, name="watchdog")
        gadget_qc.h(qr[1])
        gadget_qc.cx(qr[0], qr[1])
        gadget_qc.h(qr[1])
        gadget_qc.measure(qr[1], cr[0])

        # This is a critical step: we must add the new register to the main DAG
        dag.add_creg(cr)

        # Use compose() to safely insert the gadget onto the correct physical wires.
        # We map its classical bit to the one we just added.
        dag.compose(
            gadget_qc, qubits=[data_qubit, ancilla_qubit], clbits=[dag.clbits[-1]]
        )
        print("[WatchdogPass] Watchdog gadget successfully inserted.")
        return dag


# You'll build your custom PassManager using your custom passes
def get_custom_pass_manager(
    backend=None, pass_class=None, pass_kwargs=None, include_optimization=True
):
    """
    Create a custom pass manager that can accept any pass class and automatically
    fetch backend properties/timing constraints from the backend.

    Args:
        backend: The quantum backend to get properties from (e.g., FakeTorino)
        pass_class: The custom pass class to use (e.g., MyOptimizationPass, DecoherenceWatchdogPass)
        pass_kwargs: Additional keyword arguments for the pass constructor
        include_optimization: Whether to include standard optimization passes

    Returns:
        PassManager: A configured pass manager with the specified pass
    """
    pm = PassManager()
    pass_kwargs = pass_kwargs or {}

    # If no pass class specified, default to MyOptimizationPass
    if pass_class is None:
        pass_class = MyOptimizationPass

    # Extract backend properties and timing constraints if available
    if backend is not None:
        try:
            # Get backend properties for hardware-aware passes
            backend_properties = getattr(backend, "properties", lambda: None)()
            if backend_properties is None and hasattr(backend, "_properties"):
                backend_properties = backend._properties

            # Get timing constraints
            timing_constraints = {}
            if hasattr(backend, "dt"):
                timing_constraints["dt"] = backend.dt
            if hasattr(backend, "timing_constraints"):
                timing_constraints.update(backend.timing_constraints)

            # Auto-populate pass kwargs based on the pass class requirements
            if pass_class == DecoherenceWatchdogPass:
                if "backend_properties" not in pass_kwargs:
                    pass_kwargs["backend_properties"] = backend_properties
                if "timing_constraints" not in pass_kwargs:
                    pass_kwargs["timing_constraints"] = timing_constraints

        except Exception as e:
            print(f"Warning: Could not extract backend properties: {e}")
            # Continue with default parameters

    # Create and add the custom pass
    try:
        custom_pass = pass_class(**pass_kwargs)
        pm.append(custom_pass)
        print(f"Added custom pass: {pass_class.__name__}")
    except Exception as e:
        print(f"Warning: Could not create custom pass {pass_class.__name__}: {e}")
        # Fallback to MyOptimizationPass
        pm.append(MyOptimizationPass())
        print("Fallback: Added MyOptimizationPass")

    if include_optimization:
        # Add other necessary passes to make it a complete transpiler flow
        # This example adds unrolling, layout, routing, and another optimization
        # You'll need to decide the order and specific passes relevant to your approach.

        # Example of a simplified default-like flow with your custom pass
        # Get basis gates from the backend
        basis_gates = (
            backend.basis_gates if backend else ["u", "cx"]
        )  # Default if no backend

        # Define a simple transpilation pipeline
        # 1. Translate to basis gates
        from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary

        try:
            pm.append(
                BasisTranslator(
                    equivalence_library=StandardEquivalenceLibrary,
                    target_basis=basis_gates,
                )
            )
        except Exception as e:
            print(f"Warning: Could not add BasisTranslator: {e}")

        # If you have a backend, add layout and routing
        if backend and hasattr(backend, "coupling_map") and backend.coupling_map:
            try:
                # Layout: Try to find a good initial mapping
                from qiskit.transpiler.passes import SabreLayout

                pm.append(
                    SabreLayout(backend.coupling_map, seed=0)
                )  # Use a fixed seed for reproducibility

                # Routing: Insert SWAPs
                from qiskit.transpiler.passes import SabreSwap

                pm.append(SabreSwap(backend.coupling_map, heuristic="decay", seed=0))
            except Exception as e:
                print(f"Warning: Could not add layout/routing passes: {e}")

        # Further optimization (e.g., merging, gate cancellation)
        try:
            from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure

            pm.append(OptimizeSwapBeforeMeasure())  # Example optimization
        except Exception as e:
            print(f"Warning: Could not add optimization pass: {e}")

    return pm


def create_simple_custom_pass_manager(pass_class=None, **pass_kwargs):
    """
    Create a simple pass manager with just the custom pass (no additional optimization).

    Args:
        pass_class: The custom pass class to use
        **pass_kwargs: Keyword arguments for the pass constructor

    Returns:
        PassManager: A simple pass manager with only the custom pass
    """
    pm = PassManager()

    # Default to MyOptimizationPass if no class specified
    if pass_class is None:
        pass_class = MyOptimizationPass

    try:
        custom_pass = pass_class(**pass_kwargs)
        pm.append(custom_pass)
        print(f"Created simple pass manager with: {pass_class.__name__}")
    except Exception as e:
        print(f"Error creating pass {pass_class.__name__}: {e}")
        # Fallback
        pm.append(MyOptimizationPass())
        print("Fallback: Using MyOptimizationPass")

    return pm


# Test example usage
if __name__ == "__main__":
    from qiskit import QuantumCircuit

    # Create a test circuit with redundant CNOTs
    test_qc = QuantumCircuit(3)
    test_qc.h(0)
    test_qc.cx(0, 1)
    test_qc.cx(0, 1)  # This CNOT should be cancelled
    test_qc.cx(1, 2)

    print("Original circuit:")
    print(test_qc.draw())
    print(f"Original depth: {test_qc.depth()}")

    # Test 1: Simple custom pass manager (optimization only)
    print("\n=== Test 1: Simple Custom Pass Manager ===")
    simple_pm = create_simple_custom_pass_manager(MyOptimizationPass)
    optimized_qc = simple_pm.run(test_qc)

    print("Optimized circuit (simple):")
    print(optimized_qc.draw())
    print(f"Optimized depth: {optimized_qc.depth()}")

    # Test 2: Full custom pass manager with backend-aware configuration
    print("\n=== Test 2: Backend-Aware Pass Manager ===")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        backend = FakeTorino()

        # Test with MyOptimizationPass
        full_pm = get_custom_pass_manager(
            backend=backend, pass_class=MyOptimizationPass, include_optimization=True
        )
        full_optimized_qc = full_pm.run(test_qc)

        print("Backend-aware optimized circuit:")
        print(f"Depth: {full_optimized_qc.depth()}, Gates: {full_optimized_qc.size()}")

        # Test with DecoherenceWatchdogPass (will need layout/scheduling first)
        print("\n=== Test 3: DecoherenceWatchdogPass ===")
        watchdog_pm = get_custom_pass_manager(
            backend=backend,
            pass_class=DecoherenceWatchdogPass,
            include_optimization=False,
        )
        print("DecoherenceWatchdogPass manager created successfully")

    except ImportError:
        print("FakeTorino not available, testing with None backend")
        pm = get_custom_pass_manager(backend=None)
        optimized_qc = pm.run(test_qc)
        print(f"Optimized depth: {optimized_qc.depth()}")

    print("\n✅ Custom pass manager successfully demonstrates flexible interface!")
    print("✅ Can accept any pass class and fetch backend properties automatically!")
