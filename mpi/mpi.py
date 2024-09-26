import cudaq
from cudaq import spin
import numpy as np
from mpi4py import MPI
import time 
import cupy as cp

cudaq.set_target("nvidia")

communicator = MPI.COMM_WORLD
total_ranks = communicator.Get_size()
rank = communicator.Get_rank()

if rank == 0:
    start = time.time()

#by the time the code gets to this part, the slurm script has already distributed
#the number of tasks according to the number of tasks per node specified. 

gpus_count = cp.cuda.runtime.getDeviceCount()  #this outputs gpus on a node

#round robin distribution of ranks on a node with gpus on a node 
cp.cuda.runtime.setDevice(rank % gpus_count) 

device_id = cp.cuda.runtime.getDevice()
device_properties = cp.cuda.runtime.getDeviceProperties(device_id)
unique_gpu_identifier = device_properties['uuid'] 

qubit_count = 10
sample_count = 10000
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
split_params = communicator.scatter(params, root = 0)

      
print('total_ranks: ', total_ranks, 
      ', current rank: ', rank,
      ', unique_gpu_identifier: ', unique_gpu_identifier[:1], 
      ', params shape:', split_params.shape)


# Each process performs its work        
local_results = cudaq.observe(kernel, hamiltonian, split_params)
local_exp_vals = [result.expectation() for result in local_results]

# # Gather results from all processes
results = communicator.gather(local_exp_vals, root=0)

if rank == 0:
    final_result = np.concatenate(results)
    end = time.time()    
    print('total time', end - start )