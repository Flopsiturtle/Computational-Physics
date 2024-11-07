# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:37:15 2024

@author: Mickey
"""

"""Strang-Splitting-Integrator"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


N = 3000
epsilon = 1
mu = 1
initial = np.random.rand(N)
initial = initial/np.sqrt((np.sum(initial**2)))



def Exponential_potential(psi_in, time_step):
    shape = np.shape(psi_in)
    squares = np.zeros(shape)
    for index, value in np.ndenumerate(psi_in):
        squares[index] = np.dot(np.array(index)-int(N/2) , np.array(index)-int(N/2))
    potential_term = mu/8 * (epsilon**2 * squares - 1)**2
    psi_out = np.exp(-1j*time_step/2 * potential_term)
    psi_out = np.multiply(psi_out, psi_in)
    return psi_out

def Exponential_kinetic(psi_in, time_step):
    shape = np.shape(psi_in)
    eigenvalues = np.zeros(shape)
    D = len(shape)
    for index, value in np.ndenumerate(psi_in):
        for i in range(D):
            eigenvalues[index] += 2/(mu*epsilon**2)*(np.sin(np.pi/N*index[i]))**2
    psi_out = np.multiply(np.exp(-1j*time_step * eigenvalues), psi_in)
    return psi_out

def Strang_Splitting(psi_in, time, step_number, all_values):
    tau = time/ step_number
    wavefunctions = [initial]
    for i in range(step_number-1):
        psi_out = Exponential_potential(psi_in, tau)
        psi_out = sp.fft.fftn(psi_out)
        psi_out = Exponential_kinetic(psi_out, tau)
        psi_out = sp.fft.ifftn(psi_out)
        psi_out = Exponential_potential(psi_out,tau)
        psi_in = psi_out
        wavefunctions.append(psi_in)
    if all_values == True:
        return wavefunctions
    else:
        return wavefunctions[-1]

def test_unitarity(psi_in, time, step_number):
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = phi/np.sqrt(np.dot(np.conjugate(phi),phi))
    wavefunctions = Strang_Splitting(phi, time, step_number, True)
    for i in range(step_number):
        norm = np.dot(np.conjugate(wavefunctions[i]), wavefunctions[i])
        print(np.abs(1-norm))
    
        
        
        
    

test_unitarity(initial,2,10)    




    
    
    
    

    
    
    
    
    
    
    
    
    
    
    