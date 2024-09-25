from mpi4py import MPI
import cupy as cp


communicator = MPI.COMM_WORLD
total_ranks = communicator.Get_size()
rank = communicator.Get_rank()

gpusCount = cp.cuda.runtime.getDeviceCount()
cp.cuda.runtime.setDevice(rank % gpusCount)  #round robin distribution 
device_id = cp.cuda.runtime.getDevice()

device_props = cp.cuda.runtime.getDeviceProperties(device_id)
pci_bus_id = device_props['pciBusID']  # PCI Bus ID
name = device_props['name']  # Device name


if rank == 0:
    print("total_ranks: ", total_ranks)
    print("Total number of gpus: ", gpusCount)


print('Current rank:', rank, 
      'current device id:', device_id, 
      'pci id', cp.cuda.Device().pci_bus_id,
      'bus id', pci_bus_id,
      'name', name)

