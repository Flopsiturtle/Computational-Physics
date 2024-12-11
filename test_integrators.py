import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
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




''' --- do the tests for different dimensions and values of N, and visualize in a tabular --- '''
iterations = 10
grids = np.array([5, 10, 15,20])


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

print(' ')

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

print(' ')

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

print(' ')

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

print(' ')

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

print(' ')

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

# we also tested all these for 4 dimensions but for N=10 and beyond the computation time was just way too long






''' --- testing dependences on M and tau --- '''

####### feedback from him: maybe run rel() in other function because so much time so wen can run these tests seperately


def rel():
    global M, tau
    M_save = M
    tau_save = tau
    Ms = np.linspace(10, 1000, 100).astype(int)
    E_so = []
    E_st = []
    norm_so = []
    avg_diff = []
    for m in Ms: # change back to range 100
        M = m
        tau = 1/M # using T=1, W=1
        so = integrators.so_integrator(Psi, M,tau)
        st = integrators.Strang_Splitting(Psi, M,tau)
        E_so.append(variables.inner_product(so, hamiltonian.hamilton(so)).real)
        E_st.append(variables.inner_product(st, hamiltonian.hamilton(st)).real)
        norm_so.append(variables.inner_product(so, so).real)
        avg_diff.append(np.average(np.abs(so-st)))
        print(str(M) + " out of " + str(Ms[-1]))
    M = M_save
    tau = tau_save
    
    figure, axs = plt.subplots(2,2)
    axs[0,0].plot(Ms, np.array(E_so)/np.array(norm_so), label=r'$\frac{\langle\hat{\Psi}_{so}|\hat{H}|\hat{\Psi}_{so}\rangle}{\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle}$')  #,   
    axs[0,1].plot(Ms, E_st, label=r'$\langle\hat{\Psi}_{st}|\hat{H}|\hat{\Psi}_{st}\rangle$')
    axs[1,0].plot(Ms, norm_so, label=r'$\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle$')
    axs[1,1].plot(Ms, avg_diff, label="avg($|\hat{\Psi}_{so}-\hat{\Psi}_{st}|$)")
    for ax in axs.flat:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=28)
    
    return Ms, E_so, E_st, norm_so, avg_diff

n, Psi = variables.gaussian_1D(-int(N/4),int(N/16))
Psi = variables.normalize(Psi)

Ms, E_so, E_st, norm_so, avg_diff = rel()  

plt.show()




exit()

""" testing from above displayed without tabulars"""
iterations = 10
grids = np.array([5, 10,15])
for i in range(len(grids)):
    NN = grids[i]
    dimensions = np.array([NN,(NN,NN),(NN,NN,NN)], dtype=object)
    for j in dimensions:
        psi = np.zeros(j)
        print('"lattice" size: ' + str(j))
        print("testing unitarity of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_unitarity(integrators.so_integrator, psi, iterations)))))
        print("testing unitarity of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_unitarity(integrators.Strang_Splitting, psi, iterations)))))
        print("testing linearity of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_linearity_integrator(integrators.so_integrator, psi, iterations)))))
        print("testing linearity of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_linearity_integrator(integrators.Strang_Splitting, psi, iterations)))))
        print("testing energy conservation of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_energy_conserv(integrators.so_integrator, psi, iterations)))))
        print("testing energy conservation of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_energy_conserv(integrators.Strang_Splitting, psi, iterations)))))
