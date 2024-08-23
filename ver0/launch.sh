export LD_LIBRARY_PATH=$SCRATCH/cudaq-perlmutter/distributed_interfaces:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=$SCRATCH/cudaq-perlmutter/distributed_interfaces/libcudaq_distributed_interface_mpi.so
python3 $1

