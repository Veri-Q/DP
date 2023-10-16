import cirq
import qiskit
from qiskit.quantum_info import Operator

def qiskit2cirq(qiskit_circuit):
    conversion_rules = {
        "cx": lambda qs, params: cirq.CX(qs[0],qs[1]),
        "x": lambda qs, params: cirq.X(qs[0]),
        "y": lambda qs, params: cirq.Y(qs[0]),
        "z": lambda qs, params: cirq.Z(qs[0]),
        "h": lambda qs, params: cirq.H(qs[0]),
        "s": lambda qs, params: cirq.S(qs[0]),
        "t": lambda qs, params: cirq.T(qs[0]),
        "rx": lambda qs, params: cirq.Rx(rads=params[0])(qs[0]),
        "ry": lambda qs, params: cirq.Ry(rads=params[0])(qs[0]),
        "rz": lambda qs, params: cirq.Rz(rads=params[0])(qs[0]),
        "cy": lambda qs, params: cirq.ControlledGate(cirq.Y)(qs[0],qs[1]),
        "cz": lambda qs, params: cirq.CZ(qs[0])(qs[0],qs[1]),
        "ch": lambda qs, params: cirq.ControlledGate(cirq.H)(qs[0],qs[1]),
        "swap": lambda qs, params: cirq.SWAP(qs[0],qs[1]),
        "ccx": lambda qs, params: cirq.CCX(qs[0],qs[1],qs[2]),
        "crx": lambda qs, params: cirq.ControlledGate(cirq.Rx(rads=params[0]))(qs[0],qs[1]),
        "cry": lambda qs, params: cirq.ControlledGate(cirq.Ry(rads=params[0]))(qs[0],qs[1]),
        "crz": lambda qs, params: cirq.ControlledGate(cirq.Rz(rads=params[0]))(qs[0],qs[1]),
    }

    cirq_circuit = cirq.Circuit()
    num_qubits = qiskit_circuit.num_qubits
    cirq_qubits = cirq.GridQubit.rect(1, num_qubits)

    for gate, qubits, _ in qiskit_circuit:
        name = gate.name
        qs = [cirq_qubits[j] for j in [q.index for q in qubits]]
        params = gate.params
        
        if name in conversion_rules.keys():
            cirq_circuit.append(conversion_rules[name](qs, params))
        else:
            cirq_circuit.append(cirq.MatrixGate(Operator(gate).data)(*qs))
        
    return cirq_circuit
