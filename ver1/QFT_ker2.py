#!/usr/bin/env python3
# derived from https://nvidia.github.io/cuda-quantum/latest/examples/python/tutorials/quantum_fourier_transform.html
import cudaq
import numpy as np
from typing import List
import time

cudaq.set_target("nvidia", option="mgpu")
num_ranks = cudaq.mpi.num_ranks()
my_rank=cudaq.mpi.rank()
assert num_ranks%4==0

# util functions:
is_power_of_two = lambda n: n > 0 and (n & (n - 1)) == 0
bits_needed = lambda x: x.bit_length() if x != 0 else 1

assert is_power_of_two(num_ranks)
num_qubit = bits_needed(num_ranks)+31

shots = 10000

if my_rank==0:
    print('QFT: nq=%d numRank=%d'%(num_qubit,num_ranks))

# Define kernels for the Quantum Fourier Transform and the Inverse Quantum Fourier Transform
@cudaq.kernel
def quantum_fourier_transform2(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the QFT.
    '''
    qubit_count = len(qubits)
    # Apply Hadamard gates and controlled rotation gates.
    for i in range(qubit_count):
        h(qubits[i])
        for j in range(i + 1, qubit_count):
            angle = (2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qubits[j]], qubits[i])

@cudaq.kernel
def inverse_qft(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the inverse QFT.'''
    cudaq.adjoint(quantum_fourier_transform2, qubits)

@cudaq.kernel
def verification_circ(input_state : List[int]):
    '''Args:
    input_state (list[int]): specifies the input state to be transformed with QFT and the inverse QFT.  '''
    qubit_count = len(input_state)
    # Initialize qubits.
    qubits = cudaq.qvector(qubit_count)

    # Initialize the quantum circuit to the initial state.
    for i in range(qubit_count):
        if input_state[i] == 1: x(qubits[i])

    # Apply the quantum Fourier Transform
    quantum_fourier_transform2(qubits)

    # Apply the inverse quantum Fourier Transform
    #inverse_qft(qubits)

# The state to which the QFT operation is applied to. The zeroth element in the list is the zeroth qubit.
input_state = [0]*num_qubit

if my_rank==0 and num_qubit<12:
    print(cudaq.draw(verification_circ, input_state))
t = time.time()
counts =cudaq.sample(verification_circ, input_state, shots_count=shots)
t2 = time.time() - t
cudaq.mpi.finalize()

if my_rank==0:
    #counts.dump()
    num_sol=len(counts)
    print('adj-gpu QFT nq=%d  shots=%d  numRank=%d  num_sol=%d  elaT= %.1f sec'%(num_qubit,shots,num_ranks,num_sol, t2))
    for i,res in enumerate(counts):
        print('sol=%d %s'%(i,res))
        if i>4: break
    print('M0:done')
              

