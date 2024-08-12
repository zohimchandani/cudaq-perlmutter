On terminal in Perlmutter, run the following commands: 

1. Pull the latest image:

`shifterimg pull nvcr.io/nvidia/nightly/cuda-quantum:latest`

2. Enter the image to add some configuration:

`shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest --module=cuda-mpich /bin/bash`

3. Copy over the distributed_interfaces folder: 

`cp -r /opt/nvidia/cudaq/distributed_interfaces/ .`

4. Exit the image: 

`exit`

5. Activate the native MPI plguin 

Assuming the default MPICH loaded in Perlmutter matches the one injected into Shifter containers via the cuda-mpich module.

```
export MPI_PATH=$MPICH_DIR
source distributed_interfaces/activate_custom_mpi.sh
```

IMPORTANT:

At the time of writing, cuda-mpich shifter module is using libmpi.so from GNU 9.1 compiler, i.e., in the injected opt/udiImage/modules/cuda-mpich/lib64, libmpi.so <-> libmpi_gnu_91.so. However, gcc/9.1.0 is no longer supported as a module. Available: gcc-native/12.3, gcc/10.3.0, gcc/11.2.0,  gcc/12.2.0. MPICH for gcc-9.1 is available at /opt/cray/pe/mpich/8.1.27/ucx/gnu/9.1 although this is no longer listed in module avail.

Until the cuda-mpich shifter module is updated to match the default MPICH of Perlmutter or some modules that we can load, this is the workaround:

`export MPI_PATH=/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1`
`source distributed_interfaces/activate_custom_mpi.sh`

6. Verify the successful creation of the local library and environment variable:

`echo $CUDAQ_MPI_COMM_LIB`

7. Shifter into the container again and copy some files: 

`shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest --module=cuda-mpich /bin/bash`
`cp /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.8.89 ~/libcudart.so`
`exit` 


Now you have all the settings required to run CUDA-Q on multiple nodes on Perlmutter. 

Create a `.py` file you would like to execute and a `jobscript.sh` file. See examples of these in the repo. 

Execute the job via `sbatch jobscript.sh` on terminal. 

The output will be a `slurm-jobid.out` file, an example of which is also in the repo. 