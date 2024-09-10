import cudaq
from mpi4py import MPI
import numpy as np 
import json 
import time 

cudaq.set_random_seed(33)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()           # Which GPU to run on 
num_ranks = comm.Get_size()      # Total number of GPUs 

cudaq.set_target("nvidia")
 
qubit_count = 5
term_count = 10000
num_of_gpus = num_ranks 

if rank == 0:    # Run the code in the if statement only on the 0th GPU 
    
    hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

    batched_h = hamiltonian.distribute_terms(num_of_gpus)
    assert len(batched_h) == num_of_gpus

    serialized_batched_h = [batched_h[i].serialize() for i in range(len(batched_h))]
    assert len(serialized_batched_h) == len(batched_h)


    #check to see if broken down hamiltonian == original hamiltonian 
    reconstruct_h = cudaq.SpinOperator(serialized_batched_h[0], qubit_count)

    for h in serialized_batched_h[1:]:
        reconstruct_h += cudaq.SpinOperator(h, qubit_count)

    assert reconstruct_h == hamiltonian


    # Save to a file
    with open('serialized_batched_h.json', 'w') as f:
        json.dump(serialized_batched_h, f)


kernel = cudaq.make_kernel()
qubits = kernel.qalloc(qubit_count)
kernel.h(qubits)
kernel.x(qubits)
kernel.y(qubits)
kernel.z(qubits)


if rank == 0: 
    # Load from the file
    with open('serialized_batched_h.json', 'r') as f:
        serialized_batched_h = json.load(f)
    
    assert len(serialized_batched_h) == num_ranks
    
    #reconstruct hamiltonian 
    hamiltonian = cudaq.SpinOperator(serialized_batched_h[0], qubit_count)

    for h in serialized_batched_h[1:]:
        hamiltonian += cudaq.SpinOperator(h, qubit_count)

    #run on single GPU to compare with multi-GPU result later
    single_rank_result = cudaq.observe(kernel, hamiltonian).expectation()


else: 
    serialized_batched_h = None 
    
    
if rank == 0: 
    start = time.time()
    
# print("Rank:", rank, " Size:", num_ranks)
    
print(1)     

serialized_batched_h = comm.scatter(serialized_batched_h, root = 0) 

print(2)

batched_h = cudaq.SpinOperator(serialized_batched_h, qubit_count)

batched_result = cudaq.observe(kernel, batched_h).expectation()

results = comm.gather(batched_result, root=0)



# if rank == 0:
    
#     exp_val = sum(results)  # distributed result 
    
#     stop = time.time()
    
#     final = stop - start 
#     print('time taken with', num_of_gpus, 'gpus:', final)
    
#     print(exp_val, single_rank_result)
    
#     print(np.isclose(single_rank_result, exp_val))




