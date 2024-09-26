from mpi4py import MPI
import cupy as cp

communicator = MPI.COMM_WORLD
total_ranks = communicator.Get_size()
rank = communicator.Get_rank()

#by the time the code gets to this part, the slurm script has already distributed
#the number of tasks according to the number of tasks per node specified. 

gpus_count = cp.cuda.runtime.getDeviceCount()  #this outputs gpus on a node

#round robin distribution of ranks on a node with gpus on a node 
cp.cuda.runtime.setDevice(rank % gpus_count)  

device_id = cp.cuda.runtime.getDevice()
device_properties = cp.cuda.runtime.getDeviceProperties(device_id)
unique_gpu_identifier = device_properties['uuid'] 


print('total_ranks: ', total_ranks, 
      'current rank: ', rank,
      'unique_gpu_identifier: ', unique_gpu_identifier[:4])

# if user picks 2 nodes, 8 gpus, 16 tasks, 8 tasks/ node, 
#then 8 unique_gpu_identifier will be printed for 8 gpus and each one would be repeated twice. 