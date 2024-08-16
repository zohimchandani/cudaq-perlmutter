#!/bin/bash
#SBATCH -N 2
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -A nintern
#SBATCH -C gpu
#SBATCH --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest
#SBATCH --module=cuda-mpich

export CUDAQ_MPI_COMM_LIB=${SCRATCH}/distributed_interfaces/libcudaq_distributed_interface_mpi.so

srun -N 2 -n 8 shifter bash -l launch.sh mgpu.py
