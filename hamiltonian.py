import numpy as np

from variables import *   


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


def kinetic_hamilton(func):
    """calculating the free hamiltonian"""
    return -1/(2*mu*epsilon**2)*laplace(func)


def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*mu*epsilon**2)*laplace(func)+np.multiply(potential(func),func)


def hamilton_variable(func,mu,epsilon):     # hamilton defined in the same way as above, but with input mu and epsilon for further usage in eigenmethod tests
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*mu*epsilon**2)*laplace(func)+np.multiply(potential(func),func)