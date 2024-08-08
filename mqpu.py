import cudaq
from cudaq import spin
import time 

if cudaq.num_available_gpus() == 0:
    print("This example requires a GPU to run. No GPU detected.")

cudaq.set_target("nvidia", option="mqpu")
cudaq.mpi.initialize()

num_ranks = cudaq.mpi.num_ranks() 
rank = cudaq.mpi.rank() 

print('rank', rank, 'num_ranks', num_ranks)

if rank == 0:     
    print('mpi is initialized? ', cudaq.mpi.is_initialized())

qubit_count = 15
term_count = 100000

@cudaq.kernel
def kernel(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for i in range(1, qubit_count):
        cx(qubits[0], qubits[i])

# We create a random Hamiltonian
hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

# The observe calls allows us to calculate the expectation value of the Hamiltonian with respect to a specified kernel.

# Single node, single GPU.
t = time.time()
result0 = cudaq.observe(kernel, hamiltonian, qubit_count)
exp_val0 = result0.expectation()
t0 = time.time() - t 

# If we have multiple GPUs/ QPUs available, we can parallelize the workflow with the addition of an argument in the observe call.

# Single node, multi-GPU.
t = time.time()
result1 = cudaq.observe(kernel, hamiltonian, qubit_count, execution=cudaq.parallel.thread)
exp_val1 = result1.expectation()
t1 = time.time() - t 

# Multi-node, multi-GPU.
t = time.time()
result2 = cudaq.observe(kernel, hamiltonian, qubit_count, execution=cudaq.parallel.mpi)
exp_val2 = result2.expectation()
t2 = time.time() - t 

cudaq.mpi.finalize()

if rank == 0: 
    
    print('single gpu result', exp_val0, 'time', t0)
    print('single node multi gpu result ', exp_val1, 'time', t1)
    print('multi node multi gpu result', exp_val2, 'time', t2)



