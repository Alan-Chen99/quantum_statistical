import random
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
from axprof import AxProf
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

from Shor_Sequential_QFT import get_shor_circuit


def main():
    qc = get_shor_circuit(21, 2)
    print("ops", qc.count_ops())
    # return

    simulator = AerSimulator(method="statevector", device="GPU")
    # Aer
    circ = transpile(qc, simulator)
    res = simulator.run(circ, shots=10000).result().get_counts(qc)

    res = {int(x.split(" ")[1], 2): y for x, y in res.items()}

    return sorted(res.items())


def main2():
    print([2**i % 21 for i in range(10)])
    print([i * 128 / 6 for i in range(6)])

    # print(
    #     'Result "{0}({1})" happened {2} times out of {3}'.format(
    #         list(sim_result.get_counts().keys())[i],
    #         int(list(sim_result.get_counts().keys())[i].split(" ")[1], 2),
    #         list(sim_result.get_counts().values())[i],
    #         number_shots,
    #     )
    # )
