# mpirun -np 4 --allow-run-as-root python3 async.py
import cudaq
from cudaq import spin
import numpy as np
import time
np.random.seed(1)

cudaq.set_target("nvidia", option="mqpu")
target = cudaq.get_target()
qpu_count = target.num_qpus()
cudaq.mpi.initialize()

rank = cudaq.mpi.rank()
total_ranks = cudaq.mpi.num_ranks()

qubit_count = 20
sample_count = 1000
hamiltonian = spin.z(0)

parameters = np.random.default_rng(13).uniform(low=0,high=1,size=(sample_count, qubit_count))
split_params = np.array_split(parameters, total_ranks)

assert len(split_params) == total_ranks

if rank == 0: 
    print('target.num_qpus():', qpu_count)
    print('current rank:', rank)
    print('total ranks:', total_ranks)
    
    print('parameters shape:', parameters.shape)
    for _ in range(len(split_params)): 
        print('split parameters shape:', split_params[_].shape)
        
@cudaq.kernel
def kernel(theta:list[float]):
    qubits = cudaq.qvector(qubit_count)
    for i in range(qubit_count):
        rx(theta[i], qubits)
        
start = time.time()
result = cudaq.observe(kernel, hamiltonian, parameters)
exp_vals = [result[i].expectation() for i in range(len(result))]
end = time.time()
total = end - start 


async_start = time.time()
async_results = []
for j in range(total_ranks): 
    for i in range(split_params[j].shape[0]): 
        async_results.append(cudaq.observe_async(kernel, hamiltonian, split_params[j][i], qpu_id=j))

results = [async_results[i].get() for i in range(len(async_results))]
async_exp_vals = [results[i].expectation() for i in range(len(results))]
async_end = time.time()
async_total = async_end - async_start 

assert exp_vals == async_exp_vals

if rank == 0: 
    print('single qpu time:', total)
    print('mqpu time:', async_total)
    print('number of gpus used:', total_ranks)
    print('speedup factor', total/async_total)
    
cudaq.mpi.finalize()

