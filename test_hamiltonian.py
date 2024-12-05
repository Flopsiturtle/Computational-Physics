import numpy as np
import pandas as pd

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
    #print("the hamiltonian has been negative " + str(count) + " out of " + str(iterations) + " times")
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



''' do the tests for different dimensions and values of N '''
iterations = 10
grids = np.array([5, 10, 15])


print('Testing linearity of the hamiltonian. Maximum error: ')
tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': [], '4D': []})
for i in range(len(grids)):
    N = grids[i]
    dimensions = np.array([N,(N,N),(N,N,N),(N,N,N,N)], dtype=object)
    lst = [N]
    for j in dimensions:
        psi = np.zeros(j)
        lst.append(str(np.max(np.abs(test_linearity(hamiltonian.hamilton,psi,iterations)))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))

print('Testing hermiticity of the hamiltonian. Maximum error: ')
tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': [], '4D': []})
for i in range(len(grids)):
    N = grids[i]
    dimensions = np.array([N,(N,N),(N,N,N),(N,N,N,N)], dtype=object)
    lst = [N]
    for j in dimensions:
        psi = np.zeros(j)
        lst.append(str(np.max(np.abs(test_hermiticity(hamiltonian.hamilton, psi, iterations)))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))

print("Testing positivity of the hamiltonian. Number of times hamiltonian was negative:")
tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': [], '4D': []})
for i in range(len(grids)):
    N = grids[i]
    dimensions = np.array([N,(N,N),(N,N,N),(N,N,N,N)], dtype=object)
    lst = [N]
    for j in dimensions:
        psi = np.zeros(j)
        lst.append(str(test_positivity(hamiltonian.hamilton, psi, iterations)))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))
        

print('Testing eigenvectors of the kinetic hamiltonian. Maximum error: ')
tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': [], '4D': []})
for i in range(len(grids)):
    N = grids[i]
    dimensions = np.array([N,(N,N),(N,N,N),(N,N,N,N)], dtype=object)
    lst = [N]
    for j in dimensions:
        psi = np.zeros(j)
        lst.append(str(np.max(np.abs(test_eigenvectors(hamiltonian.kinetic_hamilton, psi, iterations)))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))




exit()

""" testing displayed without tabulars"""
iterations = 20
grids = np.array([5, 10, 15])
for i in range(len(grids)):
    N = grids[i]
    dimensions = np.array([N,(N,N),(N,N,N),(N,N,N,N)], dtype=object)
    for j in dimensions:
        psi = np.zeros(j)
        print('"lattice" size: ' + str(j))
        print("testing linearity of the hamiltonian. Maximum error: " + str(np.max(np.abs(test_linearity(hamiltonian.hamilton,psi,iterations)))))
        print("testing hermicity of the hamiltonian. Maximum error: " + str(np.max(np.abs(test_hermiticity(hamiltonian.hamilton, psi, iterations)))))
        print("testing positivity of the hamiltonian.")
        test_positivity(hamiltonian.hamilton, psi, iterations)
        print("testing eigenvectors of the kinetic hamiltonian. Maximum error: " + str(np.max(np.abs(test_eigenvectors(hamiltonian.kinetic_hamilton, psi, iterations)))))



