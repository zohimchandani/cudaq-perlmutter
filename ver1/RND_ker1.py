#!/usr/bin/env python3

# inside podman-hpc , maxNQ=33 for -N1
# salloc -N 1 --gpus-per-task=1 --ntasks-per-node=1   -t 4:00:00 -q interactive -A nintern -C gpu  --gpu-bind=none
# mpirun -np 4 python3   RND_ker1.py -q 33 -r 10 

# takes about 110 sec, inside Shifter
# srun -N 1 -n 4 shifter bash -l launch.sh " RND_ker1.py -q 32 -r 50 "
# OR RND_ker1.py --numQubits 33 --numRepeat 60  --outPath outs --expName test12
# srun -N 1 -n 1 shifter bash -l launch.sh    " RND_ker1.py -q 4 -r 4   --outPath outs --expName test12 "


import cudaq
from time import time
from pprint import pprint
import os

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=-31, type=int, help=' negative: bits in numRank - value')

    parser.add_argument('-n','--numShots', default=10000, type=int, help='num of shots')
    parser.add_argument('-r','--numRepeat', default=50, type=int, help='num of CX repeats')
    parser.add_argument('--expName', default=None, help='(optional)save summary as YAML')
    parser.add_argument('--outPath', default=None, help='(optional) for YAML')

    args = parser.parse_args()
    # util functions:
    is_power_of_two = lambda n: n > 0 and (n & (n - 1)) == 0
    bits_needed = lambda x: x.bit_length() if x != 0 else 1
    args.tStart=time()
    if 1:  # in shfter
        args.gpuOpt="mgpu,fp32"
        cudaq.set_target("nvidia", option=args.gpuOpt)        
    else: # in podman-hpc, older CudaQ version
        cudaq.set_target("nvidia-mgpu")
        cudaq.mpi.initialize() 
    args.myRank = cudaq.mpi.rank()
    args.numRanks = cudaq.mpi.num_ranks()
    
    #print('qqq',args.numQubits)
    if args.numQubits<0:
        args.numQubits=int(bits_needed(args.numRanks) - args.numQubits)
    
    if args.myRank==0:  # all ASSERT must go here
        for arg in vars(args): print( 'myArgs:',arg, getattr(args, arg))
        assert is_power_of_two(args.numRanks)
        if args.outPath and not os.path.exists(args.outPath):
            print('no outPath %s ABORT'%args.outPath); exit(99)
    assert args.numQubits>=2
    return args

    
#...!...!....................
@cudaq.kernel
def pseudoRnd_circ(qubit_count: int, nRep: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector)
    # Use poor man pseudo-random generator based on  prime numbers
    prime1, prime2 = 7919 , 104729  #  larger prime numbers
    M = qubit_count-1  # To ensure random values are within the range of qubits

    for j in range(nRep):
        for i in range(M):
            iq1= (i * prime1 + prime2) % M  # it is random index
            iq2=(iq1+1) %M
            ry(0.1*i, qvector[iq1])
            rz(0.2*i, qvector[iq2])
            x.ctrl(qvector[iq1], qvector[iq2])
    mz(qvector)

#...!...!..................
def build_meta(args):
    MD={}
    MD['num_rank']=args.numRanks
    MD['num_qubit']=args.numQubits
    MD['num_shot']=args.numShots
    MD['num_cx']=args.numRepeat*(args.numQubits-1)
    MD['cudaq_target']=args.gpuOpt
    return MD

#...!...!..................
def save_summary(args,MD):
    import yaml
    outF='nq%d_%s.yaml'%(MD['num_qubit'],args.expName)
    outFF=os.path.join(args.outPath,outF)
    ymlFd = open(outFF, 'w')
    yaml.dump(MD, ymlFd, Dumper=yaml.CDumper)
    ymlFd.close()
    print('  closed  yaml:',outFF)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    nRep=args.numRepeat
    shots=args.numShots
        
    if args.myRank==0 :
        md=build_meta(args)
        #pprint(md)
        print('adj-gpu RND nq=%d  nCX=%d  shots=%d  numRank=%d ...'%(nq,md['num_cx'],shots,args.numRanks))
        
        if md['num_cx']<30 or args.verb>1:
            print(cudaq.draw(pseudoRnd_circ, nq, nRep))

    # ... RUN SIMULATION on GPUs
    t = time()
    counts =cudaq.sample(pseudoRnd_circ, nq, nRep,shots_count=shots)
    t2 = time() - t
    cudaq.mpi.finalize()

    if args.myRank==0:
        #counts.dump()
        num_sol=len(counts)
        print('adj-gpu RND nq=%d  nCX=%d  shots=%d  numRank=%d  num_sol=%d  elaT= %.1f sec'%(nq,md['num_cx'],shots,args.numRanks,num_sol, t2))
        md['circ_time']=float('%.1f'%t2)
        md['tot_time']=float('%.1f'%(time()-args.tStart))
        sols={}
        for i,res in enumerate(counts):
            print('sol=%d %s'%(i,res))
            sols['sol%d'%i]=str(res)
            if i>3: break
        md['some_outputs']=sols
        if args.expName!=None:  save_summary(args,md)

        print('M0:done')
         
