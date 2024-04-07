import time
import cirq
import tensornetwork as tn
import jax
import jax.numpy as jnp
from jax import jit
from scipy.sparse.linalg import eigs, LinearOperator
from numpy.linalg import eigvalsh
from prettytable import PrettyTable

jax.config.update('jax_platform_name', 'cpu')
tn.set_default_backend("jax")

def reverse_circuit(circuit):
    new_circuit = cirq.Circuit()
    for gate in circuit[::-1]:
        new_circuit.append(gate)
    
    return new_circuit

def add_noise(circuit, op):
    new_circuit = cirq.Circuit()
    l = len(circuit)//2
    for gate in circuit[:l]:
        new_circuit.append(gate)

    for q in new_circuit.all_qubits():
        new_circuit.append(op(q))

    for gate in circuit[l:]:
        new_circuit.append(gate)

    return new_circuit

def circuit_to_tensor(circuit, all_qubits, measurement):
    '''
    convert a quantum circuit model to tensor network
    circuit: The quantum circuit written with cirq
    all_qubits: The total qubits, not only the working qubits of input circuit
    '''
    circuit = reverse_circuit(circuit)
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
        return y
    
    def mv2(v):
        nodes_set, left_edge, right_edge = circuit_to_tensor(model_circuit, qubits, jnp.eye(2)-measurement)
        node_v = tn.Node(v.reshape([2] * len(qubits)), axis_names=[edge.name for edge in left_edge])
        nodes_set.append(node_v)
        for j in range(len(qubits)):
            right_edge[j] ^ node_v[left_edge[j].name]

        y = tn.contractors.auto(nodes_set, left_edge).tensor.reshape([2 ** len(qubits)])
        return y

    return len(qubits), jit(mv1), jit(mv2)

def model_to_matrix(model_circuit, qubits, measurement):
    measurement = jnp.array(measurement)
    return circuit_to_matrix(model_circuit, qubits, measurement)


def circuit_to_matrix(circuit, all_qubits, measurement):
    circuit = reverse_circuit(circuit)
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

                nodes_set = [tn.contractors.auto(nodes_set, [left_edge[q] for q in all_qubits] + [right_edge[q] for q in all_qubits])]
                
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

                nodes_set = [tn.contractors.auto(nodes_set, [left_edge[q] for q in all_qubits] + [right_edge[q] for q in all_qubits])]
        
    return nodes_set[0].tensor.reshape([2 ** all_qnum, 2 ** all_qnum])


norm_jit = jit(jnp.linalg.norm)

def largest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("==========Evaluate largest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs'%(j, N, e, time.time() - start), end='\r')
        if ((time.time() - start0) / 60 / 60  > 5):
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
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
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs'%(j, N, 1 - e, time.time() - start), end='\r')
        if ((time.time() - start0) / 60 / 60  > 5):
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
            break
        
        e0 = e

    print('iter %d/%d, %.8f'%(j, N, 1 - e))
    print("===============================================")
    return 1 - e

def dp_eigenvalues_np(model_circuit, qubits, measurement):
    m1 = model_to_matrix(model_circuit, qubits, measurement)

    vals = eigvalsh(m1)
    return vals[0], vals[-1]

def dp_eigenvalues_tn(model_circuit, qubits, measurement):
    n, mv1, mv2 = model_to_mv(model_circuit, qubits, measurement)
    A1 = LinearOperator((2**n,2**n), matvec=mv1)
    A2 = LinearOperator((2**n,2**n), matvec=mv2)

    e2 = eigs(A1, k=1, return_eigenvectors=False)
    e1 = 1 - eigs(A2, k=1, return_eigenvectors=False)
    return e1, e2

def evaluate(circuit, measurements):
    noisy_p = [0.01,0.001,0.0001,0.00001,0.000001]
    noise_op = [cirq.depolarize, cirq.bit_flip]
    table = PrettyTable(["noise type", "noisy p", "kappa", "time"])
    dp_eigenvalues_np(circuit.with_noise(noise_op[0](p=0)), list(circuit.all_qubits()), measurements[0])
    for op in noise_op:
        for p in noisy_p:
            tstart = time.time()
            kappa = 1
            print(f"{op.__name__}, {p}")
            for m in measurements:
                e1, e2 = dp_eigenvalues_np(circuit.with_noise(op(p=p)), list(circuit.all_qubits()), m)
                kappa = max(kappa, e2/e1)
            
            table.add_row([op.__name__, p, f"{kappa:.3f}", f"{time.time() - tstart:.2f}"])
            print(f"{time.time() - tstart}")
    table.align = "l"
    table.vertical_char = '&'
    print(table)

def compute_delta(circuit, measurements, eta):
    noisy_p = [0.001, 0.01]
    noise_op = [cirq.depolarize, cirq.bit_flip]
    table = jnp.zeros([20,5])
    epsilons = [eta/10.*(j+1) for j in range(20)]
    dp_eigenvalues_np(circuit.with_noise(noise_op[0](p=0)), list(circuit.all_qubits()), measurements[0])
    for jop in range(2):
        op = noise_op[jop]
        for jp in range(2):
            p = noisy_p[jp]
            for m in measurements:
                e1, e2 = dp_eigenvalues_np(circuit.with_noise(op(p=p)), list(circuit.all_qubits()), m)
                print(e1, e2)
                for je in range(20):
                    e = epsilons[je]
                    print(eta*e2-(np.exp(e)+eta-1)*e1)
                    table[je,jop*2+jp] = max(table[je,jop*2+jp], eta*e2-(np.exp(e)+eta-1)*e1)

    print(table)