import numpy as np
from scipy import stats 


import variables
from variables import *
import hamiltonian


""" --- conjugate gradient --- """
# A x = b
# H x = v -> x = H^-1 v
# so conjugate gradient gives x as result which is our wanted quantity
# now: implement hamiltonian in function itself


def Hinv(v,tolerance,maxiters):
    D = len(v.shape)
    x0 = np.zeros((N,)*D)   # D-dimensional --- can be changed to x0 = np.zeros(N) if only works in 1D!!!
    r0 = v - hamiltonian.hamilton(x0)
    if np.max(r0) <= tolerance:    
        return x0
    p0 = r0
    for i in np.arange(1,maxiters+1):
        alpha0 = (np.vdot(r0,r0)) / (np.vdot(p0,hamiltonian.hamilton(p0)))    
        x = x0 + alpha0*p0
        r = r0 - alpha0*hamiltonian.hamilton(p0)
        if np.max(r) <= tolerance: 
            return x,i
        beta = (np.vdot(r,r)) / (np.vdot(r0,r0))
        p0 = r + beta*p0
        x0 = x
        r0 = r


####### for using the result in Arnoldi method: use Hinv()[0] because I am also returning the iteration as second argument







""" --- test the code --- """

# tests for arbitrary v
#### 1D test
v = np.ones(N)     
error = 0.00001
max_integers = 200
#print(Hinv(v,error,max_integers))


#### 1D test for complex with our gaussian
n, v=variables.gaussian_1D(-int(N/4),int(N/16))
v = variables.normalize(v)
#print(Hinv(v,error,max_integers))


# 2D test    ######## method doesnt work in 2D i think? but also does not need to work in 2D
#v = np.ones((N,N))
#print(Hinv(v,0.001,10))
