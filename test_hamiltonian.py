import numpy as np

import variables
from variables import *   
import hamiltonian




def test_linearity(Hamiltonian,psi_in,iterations):
    shape = np.shape(psi_in)
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    err = []
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = Hamiltonian(alpha[i]*psi1 + beta[i]*psi2)
        RHS = alpha[i]*Hamiltonian(psi1) + beta[i]*Hamiltonian(psi2)
        error = np.max(np.abs(LHS - RHS))
        err.append(error)
    return err



def test_hermiticity(Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    err = []
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = variables.inner_product(psi1, Hamiltonian(psi2))
        RHS = variables.inner_product(Hamiltonian(psi1), psi2)
        error = np.abs(LHS - RHS)
        err.append(error)
    return err



def test_positivity(Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    count = 0
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        #print("Potential:"  ,np.sign(variables.inner_product(psi1, np.multiply(hamiltonian.potential(psi1),psi1)).real))
        #print("Hamiltonian:" , np.sign(variables.inner_product(psi1, hamiltonian.hamilton(psi1)).real))
        if variables.inner_product(psi1, Hamiltonian(psi1))<0:
            count +=1
    print("the hamiltonian has been negative " + str(count) + " out of " + str(iterations) + " times")
    return(count)



def test_eigenvectors(Kinetic_Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    D = len(shape)
    plane_wave = np.zeros(shape, dtype=complex)
    err = []
    for i in range(iterations):
        k = np.random.randint(-N,N, size= D)
        eigenvalue = 0
        for index, value in np.ndenumerate(psi_in):
            plane_wave[index] = np.exp(2*np.pi * 1j * np.dot(np.array(index),k)/N)
        LHS = Kinetic_Hamiltonian(plane_wave)
        for i in range(len(k)):
            eigenvalue += (np.sin(np.pi/N * k[i]))**2
        RHS = 2/(mu*epsilon**2)*eigenvalue * plane_wave
        error = np.max(np.abs(LHS - RHS ))
        err.append(error)
    return err



''' use wavefunction '''

n, Psi=variables.gaussian_1D(-int(N/4),int(N/16))
Psi = variables.normalize(Psi)

iterations = 10
#print("testing linearity of the hamiltonian. Maximum error: " + str(np.max(np.abs(test_linearity(hamiltonian.hamilton,Psi,iterations)))))
#print("testing hermicity of the hamiltonian. Maximum error: " + str(np.max(np.abs(test_hermiticity(hamiltonian.hamilton, Psi, iterations)))))
#print("testing positivity of the hamiltonian.")
#test_positivity(hamiltonian.hamilton, Psi, iterations)
#print("testing eigenvectors of the kinetic hamiltonian. Maximum error: " + str(np.max(np.abs(test_eigenvectors(hamiltonian.kinetic_hamilton, Psi, iterations)))))


#Feedback:
#!!!! tests for multiple Dimensions but N does not have to be high (N=10,20,...)