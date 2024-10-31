# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:24:05 2024

@author: Mickey
"""

import numpy as np
import matplotlib.pyplot as plt

"""Constants"""

N = 10
epsilon = 1
mu = 1
initial = np.random.rand(N,N,N)


"""Calculate Hamiltonian of a single particle in a double well potential."""    
                        
def Laplacian(psi_in):
    """Calculates discrete Laplacian applied to a given Wavefunction psi_in. Returns wavefunction of same shape."""
    psi_out = np.empty_like(psi_in)
    psi_out = 1/(mu*epsilon**2) * psi_in
    for i in range(len(np.shape(psi_in))):
        psi_out += -1/(2*mu*epsilon**2)*(np.roll(psi_in,1, axis = i) + np.roll(psi_in, -1, axis = i))
    return psi_out

def Potential(psi_in):
    """Calculates double-well potential(consider the grid!) and applies it to a given wavefunction psi_in. Returns wavefunction of same shape. """
    squares = np.empty_like(psi_in)
    psi_out = np.empty_like(psi_in)
    for index, value in np.ndenumerate(psi_in):
        squares[index] = np.dot(np.array(index)-int(N/2) , np.array(index)-int(N/2))
    potential_term = mu/8 * (epsilon**2 * squares - 1)**2
    psi_out = np.multiply(potential_term, psi_in)
    return psi_out
        
def Hamiltonian(psi_in):
    """Calculates Hamiltonian applied to a wavefunction psi_in. Returns wavefunction of same shape."""
    psi_out =  np.empty_like(psi_in)
    psi_out = Laplacian(psi_in) + Potential(psi_in)
    return psi_out

"""Hamiltonian Tests"""

def test_linearity(Hamilton_operator,psi_in,iterations, error):
    shape = np.shape(psi_in)
    truth_values = np.array([])
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = Hamiltonian(alpha[i]*psi1 + beta[i]*psi2)
        RHS = alpha[i]*Hamiltonian(psi1) + beta[i]*Hamiltonian(psi2)
        if (np.all(np.abs(LHS - RHS) < error)):
            print(True)
        else:
            print('Alarm, Hamiltonian not linear!')
            

            
     
