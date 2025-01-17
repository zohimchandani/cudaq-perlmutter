Please make sure you work in `GLOBAL $HOME` and not `$SCRACH`

On the login node (zohim@login19:~>) in Perlmutter, run the following commands:

1. Pull the latest image:

`shifterimg pull nvcr.io/nvidia/nightly/cuda-quantum:cu11-latest`


2. Enter the image to add some configuration:

`shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:cu11-latest --module=cuda-mpich /bin/bash`

3. Copy over the distributed_interfaces folder: 

`cp -r /opt/nvidia/cudaq/distributed_interfaces/ .`

3.5. Pip install any packages you would like

4. Exit the image: 

`exit`

5. Activate the native MPI plguin 

```
export MPI_PATH=/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1
source distributed_interfaces/activate_custom_mpi.sh
```

Make sure the `distributed_interfaces` folder from step 5 above is in home directory. 

6. Verify the successful creation of the local library and environment variable:

`echo $CUDAQ_MPI_COMM_LIB`

7. Shifter into the container again and copy some files: 

```
shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:cu11-latest --module=cuda-mpich /bin/bash

cp /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.8.89 ~/libcudart.so

exit
```


Now you have all the settings required to run CUDA-Q on multiple nodes on Perlmutter. 

Create a `.py` file you would like to execute and a `jobscript.sh` file. See examples of these in the repo. 

Execute the job via `sbatch jobscript.sh` on terminal. 

The output will be a `slurm-jobid.out` file, an example of which is also in the repo. 
