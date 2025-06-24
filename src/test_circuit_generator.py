import os
import qiskit.qpy as qpy
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    QFT, EfficientSU2, BellState, GHZState, QuantumVolume,
)
from qiskit_ibm_runtime.fake_provider import FakeTorino

np.random.seed(42)  # For reproducibility
