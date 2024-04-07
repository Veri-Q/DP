import cirq
import numpy as np
from qdp import evaluate
from qiskit2cirq import qiskit2cirq

from qiskit.circuit import QuantumCircuit

circuit = qiskit2cirq(QuantumCircuit.from_qasm_file("./mnist10.qasm"))


evaluate(circuit, [np.array([[0,0],[0,1]]), np.array([[1,0],[0,0]])])
