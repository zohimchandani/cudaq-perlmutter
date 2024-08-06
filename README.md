On terminal in Perlmutter, run the following commands: 

1. `shifterimg pull nvcr.io/nvidia/nightly/cuda-quantum:latest`
2. `shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest --module=cuda-mpich /bin/bash`
3. `cp -r /opt/nvidia/cudaq/distributed_interfaces/ .`
4. `exit`
5. `export MPI_PATH=/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1`
6. `source distributed_interfaces/activate_custom_mpi.sh`
7. `echo $CUDAQ_MPI_COMM_LIB`

Now you have all the settings required to run CUDA-Q on multiple nodes on Perlmutter. 

Create a `.py` file you would like to execute and a `jobscript.sh` file. See examples of these in the repo. 

Execute the job via `sbatch jobscript.sh` on terminal. 

The output will be a `slurm-jobid.out` file, an example of which is also in the repo. 