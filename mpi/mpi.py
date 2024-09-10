import cudaq
from cudaq import spin
import numpy as np
from mpi4py import MPI

cudaq.set_target("nvidia")

comm = MPI.COMM_WORLD
total_ranks = comm.Get_size()
rank = comm.Get_rank()

if rank == 0: 
    print("Total number of ranks: ", total_ranks)
    
print("Current rank: ", rank)

qubit_count = 20
sample_count = 4000
hamiltonian = spin.z(0)

@cudaq.kernel
def kernel(thetas:list[float]):
    
    qubits = cudaq.qvector(qubit_count)
    
    for i in range(qubit_count):
        rx(thetas[i], qubits[i])


if rank == 0:
    params = np.random.default_rng(13).uniform(low=0,high=1,size=(sample_count, qubit_count))
    print(f'rank {rank} has {params.shape} elements of the parameter values')
    params = np.array_split(params, total_ranks)

else: 
    params = None


# Distribute the work (from zeroth process to non-zero processes)
split_params = comm.scatter(params, root = 0)

print(f'rank {rank} has {split_params.shape} elements of the parameter values')


# Each process performs its work        
local_results = cudaq.observe(kernel, hamiltonian, split_params)
local_exp_vals = [result.expectation() for result in local_results]

# # Gather results from all processes
results = comm.gather(local_exp_vals, root=0)

if rank == 0:
    final_result = np.concatenate(results)
    print(len(final_result))