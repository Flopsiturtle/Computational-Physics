# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:33:12 2024

@author: Florian Hollants

Goal: Animate Wavefunction that starts as gaussian centered at one minimum of the potential
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#declaring variables
H_BAR = 1
M = 1
D = 2
N = 5
A = 1

shape = (N,)  # tuple of D N's to shape ndarrays correctly
origin = (0,)  # tuple of D zeros to call position in grid later
for _ in range(D-1):
    shape += (N,)
    origin += (0,)

Psi=np.arange(N**D).reshape(shape).astype('float64')

# Create the potential
V=np.empty_like(Psi)
for n, _ in np.ndenumerate(Psi):
    V[n]=np.dot(np.array(n)-int(N/2),np.array(n)-int(N/2))

def potential():
    """defines the potential"""
    V=np.empty_like(Psi)*0
    for n, _ in np.ndenumerate(Psi):
        index_arr = np.array(n)
        V[n]=MU/8*(EPSILON**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2
    return V


def laplace(func):
    """calculating the laplacian of ndarray"""
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap


def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*MU*EPSILON**2)*laplace(func)+potential()*func



