# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:24:05 2024

@author: Mickey
"""

import numpy as np
N = 400  
epsilon = 1
mu = 1
initial = np.random.rand(N,N,N)
a = np.enumerate
                             
def Hamiltonian(psi):
    m=np.zeros(np.shape(psi))
    for i in range(len(np.shape(psi))):
        m += -1/(2*mu*epsilon**2)*(np.roll(psi,1, axis = i)-2*psi + 
                                   np.roll(psi, -1, axis = i))
    m += mu/8 * (epsilon**2*squares - 1)**2 * psi
    return m
print(Hamiltonian(initial))