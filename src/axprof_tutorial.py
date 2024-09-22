import random
import sys
import time
from pathlib import Path

from axprof import AxProf

spec = """
Input list of real;
Output real;
ACC Probability over runs [ Output ] == 0.5
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

simulator = AerSimulator(method="statevector", device="GPU")
circ = transpile(qc, simulator)


def flipCoins(dummy):
    result = simulator.run(circ, shots=1).result()
    res = list(result.get_counts(qc).keys())[0]
    return int(res)


def runner(inputFileName, config):
    ipt = float(Path(inputFileName).read_text()[:-1])
    ans = flipCoins(ipt)
    return {"acc": ans, "time": 0, "space": 0}


if __name__ == "__main__":
    startTime = time.time()  # Start measuring time
    AxProf.checkProperties(
        {},
        None,
        1,
        lambda: [random.uniform(0.1, 0.9)],
        lambda config, inputNum: [],
        runner,
        spec=spec,
    )
    endTime = time.time()  # Stop measuring time
    print("Total time required for checking:", endTime - startTime, "seconds.")
