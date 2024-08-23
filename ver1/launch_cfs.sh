export CUDAQ_MY_LIB=/global/cfs/cdirs/mpccc/balewski/tmp_cudaq_bin/
export LD_LIBRARY_PATH=$CUDAQ_MY_LIB:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=$CUDAQ_MY_LIB/distributed_interfaces/libcudaq_distributed_interface_mpi.so
python3 -u  $1

