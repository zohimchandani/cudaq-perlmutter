export LD_LIBRARY_PATH=$SCRATCH:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=$SCRATCH/distributed_interfaces/libcudaq_distributed_interface_mpi.so
python3 $1

