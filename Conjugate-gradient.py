# Author: Florian Telleis

import numpy as np
from scipy import stats 

L = 60
N = 200 
A = L/N

R = 18  # length from zero-point to potential-valleys 
Mass = 0.475   # mass of point particle
W = 1   # frequency
H_BAR = 1 #actually: 6.62607015*10**(-34)    # J*s

mu = (Mass*W*R**2)/H_BAR
epsilon = A/R


def gaussian_1D(mean,sigma): 
    x_data = np.arange(-int(N/2), int(N/2)) 
    y_data = stats.norm.pdf(x_data, mean, sigma)*np.exp(-5j*x_data) 
    return x_data, y_data 

def normalize(func):
    """normalizes input function"""
    return func*1/np.sqrt(inner_product(func,func))

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
    shape = np.shape(func)
    D = len(shape)
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap

def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*mu*epsilon**2)*laplace(func)+np.multiply(V,func)



""" --- conjugate gradient --- """
# A x = b
# H x = v -> x = H^-1 v
# so conjugate gradient gives x as result which is our wanted quantity

# implement hamiltonian in function itself



def Hinv(v,tolerance,maxiters):
    D = len(v.shape)
    x0 = np.zeros((N,)*D)   # D-dimensional --- can be changed to x0 = np.zeros(N) if only works in 1D!!!
    r0 = v - hamilton(x0)
    if np.max(r0) <= tolerance:    
        return x0
    p0 = r0
    for i in np.arange(1,maxiters+1):
        alpha0 = (inner_product(np.transpose(r0),r0)) / (inner_product(np.transpose(p0),hamilton(p0)))    # do i still have to transpose like wikipedia?
        x = x0 + alpha0*p0
        r = r0 - alpha0*hamilton(p0)
        if np.max(r) <= tolerance: 
            return x,i
        beta = (inner_product(np.transpose(r),r)) / (inner_product(np.transpose(r0),r0))
        p0 = r + beta*p0
        x0 = x
        r0 = r
        #if i == maxiters:   # or enough without if statement, because output is "None"???
        #    return "Maximum iterations of " + str(maxiters) + " reached."



""" --- run the code --- """

# 1D
n, Psi=gaussian_1D(-int(N/4),int(N/16))
V = potential(Psi)
Psi = normalize(Psi)

v = np.ones(N)
print(Hinv(v,0.00001,200))


# 2D    ######## doesnt work in 2D i think?
Psi2D = np.ones((N,N))
V = potential(Psi2D)
Psi2D = normalize(Psi2D)

v = np.ones((N,N))
print(Hinv(v,0.9,100))
