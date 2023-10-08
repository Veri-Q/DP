# DP: Tool for Detecting Violations of Differential Privacy for Quantum Algorithms

This repository contains two parts:
- An implementation for computing the condition number of a quantum decision model (See Algorithm 2 in the paper).
- Experiment codes and data for evaluation (See Section 5 in the paper).

## Requirements ##

- [Python3.8](https://www.python.org/).
- Python libraries: 
    * [Cirq](https://quantumai.google/cirq) for representing (noisy) quantum circuits.
    * [Tensornetwork](https://github.com/google/tensornetwork) for manipulating tensor networks.
    * [Numpy](https://numpy.org/) for linear algebra computations.
    * [Jax](https://github.com/google/jax) for just-in-time (JIT) compilation in Python.
    * [Qiskit](https://qiskit.org/) for manipulating quantum circuits.

## Installation (for Linux) ##

We recommend the users to use [Conda](https://docs.conda.io/en/latest/) to configure the Python environment.

### Install with Conda (Miniconda) ###
1. Follow the instructions of [Miniconda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Miniconda.
2. Clone this repository and cd to it.
    ```bash
    git clone https://github.com/Veri-Q/DP.git && cd DP
    ```
3. Use Conda to create a new Conda environment:
    ```bash
    conda create -n QDP python=3.8.12
    ```
4. Activate the above environment and use pip to install required libraries in `requirements.txt`.
    ```bash
    conda activate QDP
    pip install -r requirements.txt
    ```

## Computing the Condition Number ##

The file `qkappa.py` in this repository is the implementation of Algorithm 1 in the paper. It provides a function `kappa` that accepts a quantum decision model and outputs the model's condition number as defined in the paper. The usage of `kappa` in Python is as follows:
```python
from qkappa import kappa

# ...


k, e1, e2 = kappa(model_circuit, qubits, measurement)
# model_circuit: the (noisy) quantum circuit descried by Cirq; It expresses the super-operator $\mathcal{E}$ in the quantum decision model.
# qubits: all (cirq) qubits used in the model; usually, qubits = model_circuit.all_qubits()
# measurement: a single qubit measurement (2x2 Hermitian matrix) on the last one of all qubits in the model; It expresses the measurement $M$ at the end of the model.
# return value: k is the condition number of the model; e1 and e2 are the max/min eigenvalues.
# ...
```

For example,

```python
import cirq
import numpy as np

from qkappa import kappa

qubits = cirq.GridQubit.rect(1, 1)
model_circuit = cirq.Circuit(cirq.X(qubits[0])**0.5, cirq.depolarize(0.01)(qubits[0]))
measurement = np.array([[1., 0.], [0., 0.]])

k, e1, e2 = kappa(model_circuit, qubits, measurement)

print('The condition number is ', k)
```

## Experiments ##

🟥 Notice: Due to numerical error and randomness of the method for calculating max/min eigenvalues, the results of repeated experiments may be numerically inconsistent. 

###  Scalability in the NISQ era ###

The Benchmark circuits used in the paper are included in directories (`\QAOA`, `\HFVQE` and `\inst`)
We provide a function `testFolder` to evaluate the condition number of the benchmark circuits under a given noise type and noise level.

```python
from qkappa import *
testFolder(<path>, <noise_type>, <noise_level>)
```
`<path>` is the directory of benchmark circuits;  
`<noise_type>` is the type of noise, e.g. `cirq.bit_flip`, `cirq.phase_flip`, `cirq.depolorize`.  
`<noise_level>` is the parameter in noise operator.

For example, running the following code:
```python
from qkappa import *
testFolder('./HFVQE/', cirq.bit_flip, p=0.01)
```
can generate the result of the HFVQE benchmark circuits under bit-flip noise with noise level 0.01.

---
*🟥 The server used in our experiments has 2048GB of memory. For the users who do not have a server with the same memory, you can test on a smaller number (10-15) of qubits*.
