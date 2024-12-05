import numpy as np
from scipy.fftpack import fft, ifft

from variables import *   
import hamiltonian


### changed integrators to take tau as input, to work with modularity for testing the integrators!
def so_integrator(func,M,tau):
    """solves the time dependent schrödinger equation for a given wavefunction with the second-order integrator"""
    start = func
    for m in np.arange(0,M):
        iteration = start - 1j*tau*hamiltonian.hamilton(start) - 1/2*tau**2*hamiltonian.hamilton(hamiltonian.hamilton(start))
        start = iteration
    return iteration



def Exponential_potential(psi_in, time_step):
    shape = np.shape(psi_in)
    squares = np.zeros(shape, dtype=complex)
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



def Strang_Splitting(psi_in,M,tau):
    for i in range(M):
        psi_out = Exponential_potential(psi_in, tau)
        psi_out = fft(psi_out)
        psi_out = Exponential_kinetic(psi_out, tau)
        psi_out = ifft(psi_out)
        psi_out = Exponential_potential(psi_out,tau)
        psi_in = psi_out
    return psi_out

