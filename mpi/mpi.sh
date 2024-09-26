#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --nodes=2            
#SBATCH --ntasks=16           ## number of MPI tasks/ ranks 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:10:00
#SBATCH --qos=debug
#SBATCH --account=m4642
#SBATCH --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest
#SBATCH --module=cuda-mpich

export LD_LIBRARY_PATH=$HOME:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so
 
srun --mpi=pmix shifter bash -l launch.sh mpi.py