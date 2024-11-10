# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:37:15 2024

@author: Mickey
"""

"""Strang-Splitting-Integrator"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
%matplotlib qt



L = 5
N = 300
A = L/N
r = L/4
epsilon = A/r
mu = 6/1
centre = 100
sigma = 10
grid = np.array([i for i in range(N)])
initial = np.exp(-(grid-centre)**2/(2*sigma**2))
#initial = initial/np.sqrt((np.dot(initial,initial)))



def discrete_norm(psi_in):
    return np.multiply(np.conjugate(psi_in), psi_in)


def Potential_values(psi_in):
    """Calculates double-well potential(consider the grid!) and 
    applies it to a given wavefunction psi_in. Returns wavefunction of same shape. """
    shape = np.shape(psi_in)
    squares = np.zeros(shape)
    for index, value in np.ndenumerate(psi_in):
        squares[index] = np.dot(np.array(index)-int(N/2) , np.array(index)-int(N/2))
    potential_term = mu/8 * (epsilon**2 * squares - 1)**2
    return potential_term


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

def Strang_Splitting(psi_in, tau, step_number, all_values):
    wavefunctions = [initial]
    for i in range(step_number-1):
        psi_out = Exponential_potential(psi_in, tau)
        psi_out = sp.fft.fftn(psi_in)
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

a = Strang_Splitting(initial, 0.01, 1000, True)
fig, ax = plt.subplots()
#ax.set_xlim(-L/2,L/2)
#ax.set_ylim(-0.1,1)
line2 = ax.plot(grid, discrete_norm(initial))[0]
plt.plot(grid, Potential_values(initial))
def update(i):
    running = a[i]
    line2.set_xdata(grid)
    line2.set_ydata(discrete_norm(running))
    return line2
animation = FuncAnimation(fig, func = update, frames = 1000,interval = 10)
plt.show()

        
        
    

#test_unitarity(initial,2,10)    




    
    
    
    

    
    
    
    
    
    
    
    
    
    
    