import cirq
import numpy as np

from qkappa import kappa

qubits = cirq.GridQubit.rect(1, 1)
model_circuit = cirq.Circuit(cirq.X(qubits[0])**0.5, cirq.depolarize(0.01)(qubits[0]))
measurement = np.array([[1., 0.], [0., 0.]])

k, e1, e2 = kappa(model_circuit, qubits, measurement)

print('The condition number is ', k)