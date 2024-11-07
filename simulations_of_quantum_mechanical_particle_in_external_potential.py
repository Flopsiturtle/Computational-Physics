# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:33:12 2024

@author: Florian Hollants
"""
import numpy as np
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




def laplace(func):
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap/A**2


print(laplace(Psi))
