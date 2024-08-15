#to execute on gorby
#CUDA_VISIBLE_DEVICES=0,1 mpiexec -np 2 cudaq-qpud --port 30001 &
#CUDA_VISIBLE_DEVICES=2,3 mpiexec -np 2 cudaq-qpud --port 30002 &
#python3 remotemqpu.py
#kill %1
#kill %2

import cudaq
from cudaq import spin
import numpy as np
np.random.seed(1)

backend = 'nvidia-mgpu'
# servers = '2'  #this is equal to the number of qpus  
servers = "localhost:30001,localhost:30002"

cudaq.set_target("remote-mqpu",
                    backend=backend,
                    auto_launch=str(servers) if servers.isdigit() else "",
                    url="" if servers.isdigit() else servers)

#number of qpus = number of launched server instances 
qpu_count = cudaq.get_target().num_qpus()
qubit_count = 32
sample_count = 10
hamiltonian = spin.z(0)

parameters = np.random.default_rng(13).uniform(low=0,high=1,size=(sample_count, qubit_count))
split_params = np.array_split(parameters, qpu_count)
assert len(split_params) == qpu_count

print('number of qpus:', qpu_count)
print('qubit count', qubit_count)

print('parameters shape:', parameters.shape)
for _ in range(len(split_params)): 
    print('split parameters shape:', split_params[_].shape)
        
@cudaq.kernel
def kernel(theta:list[float]):
    qubits = cudaq.qvector(qubit_count)
    for i in range(qubit_count):
        rx(theta[i], qubits)
        

async_results = []
for j in range(len(split_params)): 
    for i in range(split_params[j].shape[0]): 
        async_results.append(cudaq.observe_async(kernel, hamiltonian, split_params[j][i], qpu_id=j))

results = [async_results[i].get() for i in range(len(async_results))]
async_exp_vals = [results[i].expectation() for i in range(len(results))]

print('job successfully executed')


