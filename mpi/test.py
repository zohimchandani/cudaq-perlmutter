import cudaq
from cudaq import spin
import numpy as np
from mpi4py import MPI
import time 
import cupy as cp


communicator = MPI.COMM_WORLD
total_ranks = communicator.Get_size()
rank = communicator.Get_rank()

gpusCount = cp.cuda.runtime.getDeviceCount()
cp.cuda.runtime.setDevice(rank % gpusCount)  #round robin distribution 
device_id = cp.cuda.runtime.getDevice()


if rank == 0:
    print("total_ranks: ", total_ranks)
    print("Total number of gpus: ", gpusCount)

print("rank: ", rank, "device_id: ", device_id)
