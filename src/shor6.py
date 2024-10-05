from fractions import Fraction
from math import gcd

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit_aer import AerSimulator


def qft(qc, qubits):
    """Apply the Quantum Fourier Transform to qubits in the circuit qc."""
    n = len(qubits)
    for i in range(n):
        qc.h(qubits[i])
        for j in range(i + 1, n):
            angle = np.pi / 2 ** (j - i)
            qc.cp(angle, qubits[j], qubits[i])


def qft_dagger(qc, qubits):
    """Apply the inverse Quantum Fourier Transform to qubits in the circuit qc."""
    n = len(qubits)
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            angle = -np.pi / 2 ** (j - i)
            qc.cp(angle, qubits[j], qubits[i])
        qc.h(qubits[i])


def c_amodN(a, power, N, n):
    """Create a controlled unitary operation that multiplies by a^power mod N."""
    if gcd(a, N) != 1:
        raise ValueError(f"'a' ({a}) and N ({N}) are not co-prime.")

    n_states = 2**n  # Total states in the register
    U = np.zeros((n_states, n_states))

    for i in range(n_states):
        if i < N:
            j = (pow(a, power, N) * i) % N
            U[i][j] = 1
        else:
            # Map states >= N to themselves (identity)
            U[i][i] = 1

    U_gate = UnitaryGate(U)
    c_U_gate = U_gate.control()
    return c_U_gate


def construct_shor_circuit(N, a):
    """Construct the quantum circuit for Shor's algorithm factoring N using base a."""
    n_count = 10  # Number of counting qubits
    n = int(np.ceil(np.log2(N)))  # Number of qubits to represent N

    qc = QuantumCircuit(n_count + n, n_count)

    # Apply Hadamard gates to counting qubits
    qc.h(range(n_count))

    # Initialize the state |1> in the auxiliary register
    qc.x(n_count)

    # Apply controlled-U operations
    for q in range(n_count):
        exponent = 2**q
        c_U_gate = c_amodN(a, exponent, N, n)
        qc.append(c_U_gate, [q] + list(range(n_count, n_count + n)))

    # Apply inverse QFT to the counting qubits
    qft_dagger(qc, range(n_count))

    # Measure the counting qubits
    qc.measure(range(n_count), range(n_count))

    return qc


def main():
    N = 21
    a = 2  # Choose 'a' such that gcd(a, N) = 1

    # Run Shor's Algorithm
    qc = construct_shor_circuit(N, a)
    simulator = AerSimulator()
    circ = transpile(qc, simulator)
    result = simulator.run(circ, shots=1000).result()
    counts = result.get_counts()
    # Convert binary string keys to integers
    res = {int(k, 2): v for k, v in counts.items()}
    # Sort the results
    return sorted(res.items())
