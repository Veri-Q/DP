import argparse
import gc
import os
import time
import cirq
import tensornetwork as tn
import jax
import jax.numpy as jnp
from jax import jit
from cirq.contrib.qasm_import import circuit_from_qasm
import signal
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
from contextlib import contextmanager

class TimeoutException(Exception): pass
jax.config.update('jax_platform_name', 'cpu')
tn.set_default_backend("jax")

@contextmanager
def time_limit(seconds = 3600):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def circuit_to_tensor(circuit, all_qubits, measurement):
    '''
    convert a quantum circuit model to tensor network
    circuit: The quantum circuit written with cirq
    all_qubits: The total qubits, not only the working qubits of input circuit
    '''
    qubits = sorted(circuit.all_qubits())
    qubits_frontier = {q: 0 for q in qubits}
    left_edge = {q: 0 for q in all_qubits}
    right_edge = {q: 0 for q in all_qubits}
    all_qnum = len(all_qubits)

    nodes_set = []

    ### Measurement
    Measurement = [jnp.eye(2)] * (all_qnum - 1) + [measurement]
    for j in range(len(Measurement)):
        left_inds = f'li{0}q{all_qubits[j]}'
        right_inds = f'ri{0}q{all_qubits[j]}'
        a = tn.Node(Measurement[j], axis_names=[left_inds, right_inds])
        nodes_set.append(a)
        left_edge[all_qubits[j]] = a[left_inds]
        right_edge[all_qubits[j]] = a[right_inds]

    ### circuit
    for moment in circuit.moments:
        for op in moment.operations:
            left_start_inds = [f"li{qubits_frontier[q]}q{q}" for q in op.qubits]
            right_start_inds = [f"ri{qubits_frontier[q]}q{q}" for q in op.qubits]
            for q in op.qubits:
                qubits_frontier[q] += 1
            left_end_inds = [f'li{qubits_frontier[q]}q{q}' for q in op.qubits]
            right_end_inds = [f'ri{qubits_frontier[q]}q{q}' for q in op.qubits]

            try:
                ### unitary
                op.gate._has_unitary_()
                U = jnp.array(cirq.unitary(op).reshape((2,) * 2 * len(op.qubits)))
                U_d = jnp.array(cirq.unitary(op).conj().T.reshape((2,) * 2 * len(op.qubits)))

                b = tn.Node(U_d, axis_names=left_end_inds + left_start_inds)
                nodes_set.append(b)
                for j in range(len(op.qubits)):
                    b[left_start_inds[j]] ^ left_edge[op.qubits[j]]
                    left_edge[op.qubits[j]] = b[left_end_inds[j]]

                c = tn.Node(U, axis_names=right_start_inds + right_end_inds)
                nodes_set.append(c)
                for j in range(len(op.qubits)):
                    c[right_start_inds[j]] ^ right_edge[op.qubits[j]]
                    right_edge[op.qubits[j]] = c[right_end_inds[j]]
                
            except:
                ### noise
                noisy_kraus = jnp.array(cirq.kraus(op))
                noisy_kraus_d = jnp.array([E.conj().T for E in cirq.kraus(op)])
                
                kraus_inds = [f'ki{qubits_frontier[q]}q{q}' for q in op.qubits]
                
                d = tn.Node(noisy_kraus_d, axis_names=kraus_inds + left_end_inds + left_start_inds)
                nodes_set.append(d)
                e = tn.Node(noisy_kraus, axis_names=kraus_inds + right_start_inds + right_end_inds)
                nodes_set.append(e)
                
                for j in range(len(kraus_inds)):
                    d[kraus_inds[j]] ^ e[kraus_inds[j]]
                
                for j in range(len(op.qubits)):
                    e[right_start_inds[j]] ^ right_edge[op.qubits[j]]
                    right_edge[op.qubits[j]] = e[right_end_inds[j]]
                    
                    d[left_start_inds[j]] ^ left_edge[op.qubits[j]]
                    left_edge[op.qubits[j]] = d[left_end_inds[j]]
        
    return nodes_set, [left_edge[q] for q in all_qubits], [right_edge[q] for q in all_qubits]

def model_to_mv(model_circuit, qubits, measurement):
    measurement = jnp.array(measurement)
    def mv1(v):
        nodes_set, left_edge, right_edge = circuit_to_tensor(model_circuit, qubits, measurement)
        node_v = tn.Node(v.reshape([2] * len(qubits)), axis_names=[edge.name for edge in left_edge])
        nodes_set.append(node_v)
        for j in range(len(qubits)):
            right_edge[j] ^ node_v[left_edge[j].name]

        y = tn.contractors.auto(nodes_set, left_edge).tensor.reshape([2 ** len(qubits)])
        e = jnp.linalg.norm(y)
        return y / e, e
    
    def mv2(v):
        nodes_set, left_edge, right_edge = circuit_to_tensor(model_circuit, qubits, jnp.eye(2)-measurement)
        node_v = tn.Node(v.reshape([2] * len(qubits)), axis_names=[edge.name for edge in left_edge])
        nodes_set.append(node_v)
        for j in range(len(qubits)):
            right_edge[j] ^ node_v[left_edge[j].name]

        y = tn.contractors.auto(nodes_set, left_edge).tensor.reshape([2 ** len(qubits)])
        e = jnp.linalg.norm(y)
        return y / e, e

    return len(qubits), jit(mv1), jit(mv2)

norm_jit = jit(jnp.linalg.norm)

def largest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("==========Evaluate largest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    for j in range(N):
        gc.collect()
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs'%(j, N, e, time.time() - start), end='\r')
        if jnp.abs(e - e0) < 1e-6 and j>10:
            break
        e0 = e

    print('iter %d/%d, %.8f'%(j, N, e))
    print("===============================================")
    return e

def smallest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("=========Evaluate smallest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    for j in range(N):
        gc.collect()
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs'%(j, N, 1 - e, time.time() - start), end='\r')
        if jnp.abs(e - e0) < 1e-6 and j>10:
            break
        e0 = e

    print('iter %d/%d, %.8f'%(j, N, 1 - e))
    print("===============================================")
    return 1 - e

def kappa(model_circuit, qubits, measurement):
    n, mv1, mv2 = model_to_mv(model_circuit, qubits, measurement)
    e1 = largest_eigenvalue(n, mv1, 100)
    if e1 == -1:
        return -1
    e2 = smallest_eigenvalue(n, mv2, 100)
    if e2 == -1:
        return -1

    return e1/e2, e1, e2


def getTestCircuit(file, noise_op, p=0.01):
    with open(file, 'r') as f:
        qasm_str = f.read()
    cir = QuantumCircuit.from_qasm_str(qasm_str)
    cir.remove_final_measurements()
    cir = RemoveBarriers()(cir)
    qasm_str = cir.inverse().qasm()
    cir = circuit_from_qasm(qasm_str)
    qubits = cir.all_qubits()

    # noise_layer = [noise_op(p).on(q) for q in qubits]
    # noisy_circuit = cirq.Circuit(noise_layer)
    # noisy_circuit += cir
    # noisy_circuit = noisy_circuit.inverse()
    # qubits = sorted(noisy_circuit.all_qubits())
    circuit = circuit_from_qasm(qasm_str)
    qubits = sorted(circuit.all_qubits())
    if p > 1e-7:
        circuit += noise_op(p).on_each(*qubits)
    return qubits, circuit


def testFile(file, noise_op = cirq.depolarize, p=0.01):
    try:
        with time_limit():
            qubits, model_circuit = getTestCircuit(file, noise_op, p)
            measurement = jnp.array([[1., 0.], [0., 0.]])
            tStart = time.time()
            k, e1, e2 = kappa(model_circuit, qubits, measurement)
            totalTime = time.time()-tStart
            print('Circuit: %s'%file)
            print('Noise configuration: %s, %f'%(noise_op, p))
            print('Total execution time: %.4fs'%totalTime)
            print('Condition Number: %.6f'%k)
            print('(The max/min eigenvalues are: %.4f, %.4f)'%(e1, e2))
    except TimeoutException as e:
        print('Time out!')
    except Exception as e:
        raise

def testFolder(path, noise_op = cirq.depolarize, p=0.01):
    files = os.listdir(path)
    for f in files:
        testFile(path+f, noise_op, p)
        gc.collect()

if __name__ == '__main__':
    testFile('./QAOA/qaoa_10.qasm', cirq.bit_flip, p=0.01)
    testFolder('./HFVQE/', cirq.bit_flip, p=0.01)

    
