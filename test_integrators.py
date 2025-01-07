import numpy as np
from scipy.fftpack import fft, ifft
import pandas as pd


import hamiltonian
import integrators
import variables
from variables import *   




def test_unitarity(integrator, psi_in, iterations):
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = variables.normalize(phi)
    err = []
    for i in range(iterations):
        wave = integrator(phi,iterations,tau)   
        norm = np.sqrt(variables.inner_product(wave,wave))
        error = np.abs(1-norm)
        err.append(error)
        phi = wave
    return err

def test_linearity_integrator(integrator, psi_in, iterations):
    shape = np.shape(psi_in)
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    err = []
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = integrator((alpha[i]*psi1 + beta[i]*psi2), 1,tau)
        RHS = alpha[i]*integrator(psi1, 1,tau) + beta[i]*integrator(psi2,1,tau)
        error = np.max(np.abs(LHS - RHS))
        err.append(error)
    return err
       
    
def test_energy_conserv(integrator, psi_in, iterations): 
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = variables.normalize(phi)
    energy0 = variables.inner_product(phi, hamiltonian.hamilton(phi))
    err = []
    for i in range(iterations):
        wave = integrator(phi, iterations,tau) 
        energy0 = variables.inner_product(phi, hamiltonian.hamilton(phi))/variables.inner_product(phi, phi)
        energy1 = variables.inner_product(wave, hamiltonian.hamilton(wave))/variables.inner_product(wave, wave)
        error = np.abs(energy1 - energy0)
        err.append(error)
        phi = wave        
    return err




''' --- define different parameter-tests which output tabulars and can be called separately --- '''

def tab_unit_so(grids,iterations):
    print('Testing unitarity of the Second-Order integrator ' + str(iterations) + ' times. Maximum error: ')
    tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': []})
    for i in range(len(grids)):
        N = grids[i]
        dimensions = np.array([N,(N,N),(N,N,N)], dtype=object)
        lst = [N]
        for j in dimensions:
            psi = np.zeros(j)
            lst.append(str(np.max(np.abs(test_unitarity(integrators.so_integrator, psi, iterations)))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))

def tab_unit_ss(grids,iterations):
    print('Testing unitarity of the Strang-Splitting integrator ' + str(iterations) + ' times. Maximum error: ')
    tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': []})
    for i in range(len(grids)):
        N = grids[i]
        dimensions = np.array([N,(N,N),(N,N,N)], dtype=object)
        lst = [N]
        for j in dimensions:
            psi = np.zeros(j)
            lst.append(str(np.max(np.abs(test_unitarity(integrators.Strang_Splitting, psi, iterations)))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))

def tab_lin_so(grids,iterations):
    print('Testing linearity of the Second-Order integrator ' + str(iterations) + ' times. Maximum error: ')
    tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': []})
    for i in range(len(grids)):
        N = grids[i]
        dimensions = np.array([N,(N,N),(N,N,N)], dtype=object)
        lst = [N]
        for j in dimensions:
            psi = np.zeros(j)
            lst.append(str(np.max(np.abs(test_linearity_integrator(integrators.so_integrator, psi, iterations)))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))

def tab_lin_ss(grids,iterations):
    print('Testing linearity of the Strang-Splitting integrator ' + str(iterations) + ' times. Maximum error: ')
    tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': []})
    for i in range(len(grids)):
        N = grids[i]
        dimensions = np.array([N,(N,N),(N,N,N)], dtype=object)
        lst = [N]
        for j in dimensions:
            psi = np.zeros(j)
            lst.append(str(np.max(np.abs(test_linearity_integrator(integrators.Strang_Splitting, psi, iterations)))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))

def tab_Econserv_so(grids,iterations):
    print('Testing energy conservation of the Second-Order integrator ' + str(iterations) + ' times. Maximum error: ')
    tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': []})
    for i in range(len(grids)):
        N = grids[i]
        dimensions = np.array([N,(N,N),(N,N,N)], dtype=object)
        lst = [N]
        for j in dimensions:
            psi = np.zeros(j)
            lst.append(str(np.max(np.abs(test_energy_conserv(integrators.so_integrator, psi, iterations)))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))

def tab_Econserv_ss(grids,iterations):
    print('Testing energy conservation of the Strang-Splitting integrator ' + str(iterations) + ' times. Maximum error: ')
    tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': []})
    for i in range(len(grids)):
        N = grids[i]
        dimensions = np.array([N,(N,N),(N,N,N)], dtype=object)
        lst = [N]
        for j in dimensions:
            psi = np.zeros(j)
            lst.append(str(np.max(np.abs(test_energy_conserv(integrators.Strang_Splitting, psi, iterations)))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))



''' --- run the code --- '''
iterations = 10
grids = np.array([5, 10, 15,20])

tab_unit_so(grids,iterations)
print(' ')
tab_unit_ss(grids,iterations)
print(' ')
tab_lin_so(grids,iterations)
print(' ')
tab_lin_ss(grids,iterations)
print(' ')
tab_Econserv_so(grids,iterations)
print(' ')
tab_Econserv_ss(grids,iterations)

# we also tested all these for 4 dimensions, but the computation time was just way too long for N=10 (and beyond) 

