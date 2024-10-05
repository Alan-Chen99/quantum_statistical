from fractions import Fraction
from math import gcd

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit_aer import AerSimulator


def c_amodN(qc, q_control, q_target, a, power, N):
    """Controlled multiplication of a^power mod N"""
    # Calculate the multiplicand: a^power mod N
    multiplicand = pow(a, power, N)
    n = q_target.size  # Number of qubits in the target register

    # Create the unitary matrix for the modular multiplication
    U = np.zeros((2**n, 2**n))
    for x in range(2**n):
        # Only compute for x < N to keep the register within [0, N-1]
        if x < N:
            y = (multiplicand * x) % N
        else:
            y = x  # States representing numbers >= N map to themselves
        U[y, x] = 1  # Map |x⟩ -> |(multiplicand * x) mod N⟩

    # Convert the unitary matrix to a UnitaryGate
    mod_mult_gate = UnitaryGate(U, label=f"×{multiplicand} mod {N}")

    # Make the gate controlled
    c_mod_mult_gate = mod_mult_gate.control(1)

    # Apply the controlled modular multiplication
    qc.append(c_mod_mult_gate, [q_control] + q_target[:])


def main():

    # Number to factor
    N = 21
    a = 2

    n = N.bit_length()
    n_count = 2 * n

    # Quantum registers
    qr_count = QuantumRegister(n_count, name="count")  # Counting qubits
    qr_mult = QuantumRegister(n, name="mult")  # Multiplication qubits

    # Classical register to store the measurement
    cr = ClassicalRegister(n_count, name="classical")

    # Quantum circuit combining registers
    qc = QuantumCircuit(qr_count, qr_mult, cr)

    # Apply Hadamard gates to counting qubits
    qc.h(qr_count)

    # Initialize multiplication register to |1⟩
    qc.x(qr_mult[0])

    # Apply the controlled modular multipliers
    for i in range(n_count):
        power = 2**i
        c_amodN(qc, qr_count[n_count - i - 1], qr_mult, a, power, N)

    # Apply inverse QFT to counting qubits
    qc.append(QFT(num_qubits=n_count, inverse=True, do_swaps=False), qr_count)  # type: ignore

    # Measure counting qubits
    qc.measure(qr_count, cr)

    simulator = AerSimulator(max_shot_size=1)
    print("transpile")
    circ = transpile(qc, simulator)
    print("run")
    result = simulator.run(circ, shots=100000).result()
    counts = result.get_counts()
    res = {int(k, 2): v for k, v in counts.items()}
    return sorted(res.items())
