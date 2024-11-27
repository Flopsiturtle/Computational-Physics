# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:42:34 2024

@author: flori
"""

import numpy as np
from scipy import stats

L = 60
N = 100 # number of points in each direction of D-dimensional lattice  # if wavefunction is given: N = (wave.shape)[0]    # \\ [0] is arbitrary because quadratic matrix
A = L/N  # spacing between lattice points   # assumption: A is input | can also be variant of L=N*a  

R = 18  # length from zero-point to potential-valleys 
Mass = 0.475   # mass of point particle
W = 1   # frequency
H_BAR = 1#!!! actually: 6.62607015*10**(-34)    # J*s

mu = (Mass*W*R**2)/H_BAR
epsilon = A/R


T = 10     # time
M = 1000*T  # large value
tau = W*T/M  # time step


def gaussian_1D(mean,sigma): 
    x_data = np.arange(-int(N/2), int(N/2)) 
    y_data = stats.norm.pdf(x_data, mean, sigma)*np.exp(-5j*x_data) 
    return x_data, y_data 

def inner_product(func1, func2):
    """calculates the inner product of two arrays"""
    return np.dot(np.conjugate(func1),func2)

def potential(func):
    """defines the potential"""
    V = np.zeros(func.shape)
    for n, _ in np.ndenumerate(func):
        index_arr = np.array(n)
        V[n]=mu/8*(epsilon**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2
    return V


def laplace(func):
    """calculating the laplacian of ndarray"""
    D = len(np.shape(func))
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap


def hamilton(func, V=None):
    """calculating the hamiltonian for double harmonic well"""
    Hfunc = -1/(2*mu*epsilon**2)*laplace(func)
    if V is not None:
        Hfunc += np.multiply(V,func)
    return Hfunc


def power_method(Qfunc, v, tol=1e-6, max_iter=100000):
    """
    Generator function for the power method.
    
    Parameters:
        Qfunc (function): function to act on the state
        v (ndarray): Initial guess for the eigenvector.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    
    Yields:
        tuple: (current eigenvalue, current eigenvector)
    """
    w = v / np.linalg.norm(v)  # Normalize the initial vector
    lambda_old = 0
    
    for _ in range(max_iter):
        # Apply hamiltonian
        Qw = Qfunc(w)
        # Compute the new Energy Eigenvalue
        lambda_new = inner_product(w, Qw)  #Energy
        # Normalize the eigenvector
        w = Qw / np.linalg.norm(Qw)
        
        # generates the current estimate of eigenvalue and eigenvector
        yield lambda_new, w
        
        # Check for convergence
        if np.abs(lambda_new - lambda_old) < tol:
            print(lambda_new)
            break
        if np.linalg.norm(Qw - lambda_new*w) < tol:
            print(lambda_new)
            break
        lambda_old = lambda_new


def Hinv():
    


#----------------------------------------------------------------
n, Psi=gaussian_1D(-int(N/4),int(N/16))
V = potential(Psi)

x = [np.random.rand()+np.random.rand()*1j for _ in range(100)]
t = 0
for eigenvalue, eigenvector in power_method(hamilton, x):
    t+=1
    print(t)
