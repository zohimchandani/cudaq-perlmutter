export LD_LIBRARY_PATH=$HOME:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=$HOME/distributed_interfaces/libcudaq_distributed_interface_mpi.so
python3 $1
