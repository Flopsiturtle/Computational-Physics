# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:24:05 2024

@author: Mickey
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
epsilon = 1
mu = 1
initial = np.random.rand(N)

                             
def Laplacian(psi_in):
    """Calculates discrete Laplcian of a given Wavefunction psi_in"""
    psi_out = np.empty_like(psi_in)
    psi_out = 1/(mu*epsilon**2) * psi_in
    for i in range(len(np.shape(psi_in))):
        psi_out += -1/(2*mu*epsilon**2)*(np.roll(psi_in,1, axis = i) + np.roll(psi_in, -1, axis = i))
    return psi_out

def Potential(psi_in):
    squares = np.empty_like(psi_in)
    psi_out = np.empty_like(psi_in)
    for index, value in np.ndenumerate(psi_in):
        squares[index] = np.dot(np.array(index)-int(N/2) , np.array(index)-int(N/2))
    potential_term = mu/8 * (epsilon**2 * squares - 1)**2
    psi_out = np.multiply(potential_term, psi_in)
    return psi_out
        
def Hamiltonian(psi_in):
    psi_out =  np.empty_like(psi_in)
    psi_out = Laplacian(psi_in) + Potential(psi_in)
    return psi_out

