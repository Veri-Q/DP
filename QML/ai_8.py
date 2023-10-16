import cirq
import numpy as np
from qdp import evaluate

NUM_QUBITS = 8
WORKING_QUBITS = cirq.GridQubit.rect(1,NUM_QUBITS)

def generate_model_circuit(variables):
    qubits = WORKING_QUBITS
    symbols = iter(variables)
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
        
    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
        
    circuit += cirq.X(qubits[-1]) ** next(symbols)
    circuit += cirq.Y(qubits[-1]) ** next(symbols)
    circuit += cirq.X(qubits[-1]) ** next(symbols)
    
    return circuit

# imported from a trained model
params = [5.6864963,  4.5384674,  4.0658937,  6.0114822,  2.6314237,  0.7971049,
           6.2414956,  1.231465 ,  5.112798  , 0.09745377, 0.2654334,  4.1310773,
           3.3447504,  5.935498 ,  1.7449    , 1.745954  , 1.514159 ,  2.4577525,
           6.188601 ,  5.751889 ,  0.16371164, 5.015923  , 2.698336 ,  2.7948823,
           1.7905817,  4.1858573,  1.714581  , 4.134787  , 4.522799 ,  0.33325404,
           5.646758 ,  1.0231644,  3.535049  , 4.513359  , 2.4423301,  3.346549,
           0.7184883,  3.5541363,  5.1378045 , 5.4350505 , 4.250444 ,  2.081229,
           2.3359709,  1.1259285,  3.906016  , 0.1284471 , 2.5366719,  5.801898,
           1.9489733,  2.5943935,  5.240497  , 2.2280385 , 2.2115154,  3.0721598,
           0.9330431,  2.9257228,  2.702144  , 4.1977177 , 1.682387 ,  3.859512,
           4.688113 ,  5.4294186,  3.3565576 , 6.080049  , 1.753433 ,  1.5129646,
           5.4340334, ]

circuit = generate_model_circuit(params)

evaluate(circuit, [np.array([[0,0],[0,1]]), np.array([[1,0],[0,0]])])
