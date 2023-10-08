from qiskit import transpile, QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
from qiskit import *
from qiskit import Aer
import numpy as np

def bit_flip(rho, nqubit, target, p):
    X_t = np.array([1])
    for i in range(nqubit):
        if i == target:
            X_t = np.kron(X_t, np.array([[0, 1], [1, 0]]))
        else:
            X_t = np.kron(X_t, np.eye(2))
    return p* X_t @ rho @ X_t + (1-p)* rho

def bit_flip_validation(M, U, nqubit, p):
    mat = U.conj().T @ M @ U
    for t in range(nqubit):
       mat = bit_flip(mat, nqubit, t, p)
    eig, vec = np.linalg.eig(mat)
    return max(eig)

if __name__ == '__main__':
    nqubit = 10
    backend = Aer.get_backend('unitary_simulator')
    with open('./QAOA/qaoa_10.qasm', 'r') as f:
        qasm_str = f.read()
    cir = QuantumCircuit.from_qasm_str(qasm_str)
    cir.remove_final_measurements()
    cir = RemoveBarriers()(cir)
    job = execute(cir, backend)
    result = job.result()  
    U = result.get_unitary(cir, decimals = 10).data
    M = np.array([1])
    for i in range(nqubit-1):
        M = np.kron(M, np.eye(2))
    M = np.kron(M, np.array([[1, 0], [0, 0]]))
    print(bit_flip_validation(M, U, nqubit, 0.1))