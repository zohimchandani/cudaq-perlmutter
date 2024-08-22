#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

comm=" -t 1:00:00 -q regular"
#comm=''

#for N in 1 2 4 8 16 32 ; do
for N in 64 128 256 ; do
    echo N=$N
    sbatch -N $N $comm batch_mgpu.slr 
done
