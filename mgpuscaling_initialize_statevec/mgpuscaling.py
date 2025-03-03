# mpirun -np xx --allow-run-as-root python3 test.py --cudaq-full-stack-trace

import cudaq
import cupy as cp
import numpy as np

cudaq.set_target('nvidia', option='mgpu,fp64')
cudaq.mpi.initialize()

num_qubits = 34
num_ranks = cudaq.mpi.num_ranks()
rank = cudaq.mpi.rank()

print('current rank', rank, 'total ranks', num_ranks)

@cudaq.kernel
def alloc_kernel():
    qubits = cudaq.qvector(num_qubits)

#allocate a n qubit statevec on gpu memory taking advanatge of mgpu functionality 
alloc_state = cudaq.get_state(alloc_kernel)

@cudaq.kernel
def kernel(state: cudaq.State):
    qubits = cudaq.qvector(state)
    #put gates of interest here

#leave these imports here, do not move it to before kernel definition
from cupy.cuda.memory import MemoryPointer, UnownedMemory

def to_cupy_array(allocated_state):
    
    #obtain the allocated state that lives on gpu memory and assign it to a variable
    state_on_gpu = allocated_state.getTensor()
    
    #find memory address of the allocated state on gpu memory  
    mem_address_state_on_gpu = state_on_gpu.data() 
    
    #prints how many amplitudes live on each rank. Sum of amp per rank = 2**nqubits
    # print("Number of amplitudes on each rank =", state_on_gpu.get_num_elements()) 
    
    #calculate memory the state occupies in bytes per rank
    sizeByte = state_on_gpu.get_num_elements() * state_on_gpu.get_element_size()
    # print("size of allocated state distributed on each rank =", sizeByte) 

    #use cupy to assign the statevec memory to a cupy array 
    mem = UnownedMemory(mem_address_state_on_gpu, sizeByte, owner=allocated_state)
    
    memptr = MemoryPointer(mem, 0)
    
    cupy_array = cp.ndarray(state_on_gpu.get_num_elements(),
                              dtype=cp.complex128,
                              memptr=memptr)
    return cupy_array


def split_sv_amplitudes(num_qubits, num_ranks): 
    
    '''calculates the total number of amplitudes of a given n qubit statevec and outputs a list of evenly
    distributed amplitudes per rank'''
    
    len_statevec = 2**num_qubits  

    base = len_statevec // num_ranks  
    remainder = len_statevec % num_ranks 

    lens_of_distributed_statevec = [base + 1 if i < remainder else base for i in range(num_ranks)]

    assert sum(lens_of_distributed_statevec) == len_statevec, 'statevector not distributed correctly'
    assert len(lens_of_distributed_statevec) == num_ranks, 'not distributed correctly'
    
    return lens_of_distributed_statevec

#assign initial allocated distributed state to a cupy array on each rank 
rank_slice = to_cupy_array(alloc_state)

# #Strategy 1: define a large statevec on cpu memory and then split it 
# #initialize 2**n statvector on cpu memory and then split it 
# sv = (np.random.randn(2**num_qubits) +  1j * np.random.randn(2**num_qubits)).astype(np.complex128) 
# sv = sv/np.linalg.norm(sv) #normalize statevecor - if you dont normalize, results make no sense
# split_sv = np.array_split(sv, num_ranks) #split statevec 

# #num of splits == num of ranks 
# #num of elements in a split == num of elemments blocked on gpu memory waiting to be populated 
# assert len(split_sv) == num_ranks 
# for sub_state_vec in split_sv: 
#     assert len(sub_state_vec) == len(rank_slice)

# #copy split statevecs to the cupy arrays holding the initial allocated state  
# cp.cuda.runtime.memcpy(rank_slice.data.ptr, split_sv[rank].ctypes.data, split_sv[rank].nbytes, cp.cuda.runtime.memcpyHostToDevice)


# Strategy 2: define split statevecs to begin with on each rank 
# amplitudes per rank as a list 
amps_per_rank = split_sv_amplitudes(num_qubits, num_ranks)

split_sv = (np.random.randn(amps_per_rank[rank]) +  1j * np.random.randn(amps_per_rank[rank])).astype(np.complex128) 
sum_mod_sq_per_rank = [sum(np.abs(i)**2 for i in split_sv)]
print('current rank', rank, 'total ranks', num_ranks, 'sum_mod_sq_per_rank', sum_mod_sq_per_rank)

#gather_sums shows up on all ranks 
gather_sums = cudaq.mpi.all_gather(len(sum_mod_sq_per_rank)*cudaq.mpi.num_ranks(), sum_mod_sq_per_rank)
print('current rank', rank, 'total ranks', num_ranks, 'gather_sums', gather_sums )

#norm shows up on all ranks hence no need to broadcast
norm = [np.sqrt(np.sum(gather_sums))]
print('current rank', rank, 'total ranks', num_ranks, 'norm', norm)

# cudaq.mpi.broadcast(norm, 0, 1)

split_sv = split_sv/norm #normalize

cp.cuda.runtime.memcpy(rank_slice.data.ptr, split_sv.ctypes.data, split_sv.nbytes, cp.cuda.runtime.memcpyHostToDevice)

#execute the kernel with the initial allocated state which has now been populated with the state of interest
result = cudaq.sample(kernel, alloc_state)
# result = cudaq.get_state(kernel, alloc_state)

# print(result)

assert num_qubits == len(list(result)[0])  

print('done')

cudaq.mpi.finalize()