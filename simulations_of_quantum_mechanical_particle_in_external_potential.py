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
MASS = 1
D = 2
N = 5
A = 1
OMEGA=1
R=L/4
T=1
M=1000

mu=MASS*OMEGA*R**2/H_BAR
epsilon = A/R
tau=OMEGA*T/M


shape = (N,)  # tuple of D N's to shape ndarrays correctly
origin = (0,)  # tuple of D zeros to call position in grid later
for _ in range(D-1):
    shape += (N,)
    origin += (0,)


def gaussian_1D(mean,sigma): 
    x_data = np.arange(0, N) 
    y_data = stats.norm.pdf(x_data, mean, sigma) 
    return y_data 


def potential(func):
    """defines the potential"""
    V = np.zeros(func.shape)
    for n, _ in np.ndenumerate(func):
        index_arr = np.array(n)
        V[n]=mu/8*(epsilon**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2
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
    return -1/(2*mu*epsilon**2)*laplace(func)+V*func


Psi=gaussian_1D(25,10)
V = potential(Psi)

