# This file can be executed via: mpiexec -np 4 --allow-run-as-root python3 mgpu.py

# The mgpu target pools gpu memory together to enable scaling qubit count 

import cudaq

cudaq.set_target("nvidia", option="mgpu")

qubit_count = 31
term_count = 10

@cudaq.kernel
def kernel(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for i in range(1, qubit_count):
        cx(qubits[0], qubits[i])

hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count, seed = 44)

result = cudaq.observe(kernel, hamiltonian, qubit_count).expectation()
print(result)



