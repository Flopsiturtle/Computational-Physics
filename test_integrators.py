import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

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
        wave = integrator(phi,iterations)   
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
        LHS = integrator((alpha[i]*psi1 + beta[i]*psi2), 1)
        RHS = alpha[i]*integrator(psi1, 1) + beta[i]*integrator(psi2,1)
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
        wave = integrator(phi, iterations) 
        energy0 = variables.inner_product(phi, hamiltonian.hamilton(phi))/variables.inner_product(phi, phi)
        energy1 = variables.inner_product(wave, hamiltonian.hamilton(wave))/variables.inner_product(wave, wave)
        error = np.abs(energy1 - energy0)
        err.append(error)
        phi = wave        
    return err



''' use wavefunction '''

n, Psi = variables.gaussian_1D(-int(N/4),int(N/16))
Psi = variables.normalize(Psi)

iterations = 10
#print("testing unitarity of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_unitarity(integrators.so_integrator, Psi, iterations)))))
#print("testing unitarity of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_unitarity(integrators.Strang_Splitting, Psi, iterations)))))
#print("testing linearity of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_linearity_integrator(integrators.so_integrator, Psi, iterations)))))
#print("testing linearity of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_linearity_integrator(integrators.Strang_Splitting, Psi, iterations)))))
#print("testing energy conservation of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_energy_conserv(integrators.so_integrator, Psi, iterations)))))
#print("testing energy conservation of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_energy_conserv(integrators.Strang_Splitting, Psi, iterations)))))

#Feedback:
#!!!! tests for multiple Dimensions but N does not have to be high (N=10,20,...)


def rel():
    global M, tau
    M_save = M
    tau_save = tau
    Ms = []
    E_so = []
    E_st = []
    norm_so = []
    avg_diff = []
    for m in range(10)[1:]: # change back to range 100
        M = 10*m
        tau = 1/M # using T=1, W=1
        so = integrators.so_integrator(Psi, M)
        st = integrators.Strang_Splitting(Psi, M)
        Ms.append(M)
        E_so.append(variables.inner_product(so, hamiltonian.hamilton(so)).real)
        E_st.append(variables.inner_product(st, hamiltonian.hamilton(st)).real)
        norm_so.append(variables.inner_product(so, so).real)
        avg_diff.append(np.average(np.abs(so-st)))
        print(str(M) + " out of 990")
    M = M_save
    tau = tau_save
    
    figure, axs = plt.subplots(2,2)
    axs[0,0].set(xlabel=r'log(M)')
    axs[0,1].set(xlabel=r'log(M)')
    axs[1,0].set(xlabel=r'log(M)')
    axs[1,1].set(xlabel=r'log(M)')
    axs[0,0].plot(np.log(Ms), np.log(np.array(E_so)/np.array(norm_so)), label=r'$log\left(\frac{\langle\hat{\Psi}_{so}|\hat{H}|\hat{\Psi}_{so}\rangle}{\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle}\right)$')  #,   
    axs[0,1].plot(np.log(Ms), np.log(E_st), label=r'$log(\langle\hat{\Psi}_{st}|\hat{H}|\hat{\Psi}_{st}\rangle)$')
    axs[1,0].plot(np.log(Ms), np.log(norm_so), label=r'$log(\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle)$')
    axs[1,1].plot(np.log(Ms), np.log(avg_diff), label="log(avg($|\hat{\Psi}_{so}-\hat{\Psi}_{st}|$))")
    axs[0,0].legend(fontsize=18)
    axs[0,1].legend(fontsize=18)
    axs[1,0].legend(fontsize=18)
    axs[1,1].legend(fontsize=18)
    return Ms, E_so, E_st, norm_so, avg_diff

Ms, E_so, E_st, norm_so, avg_diff = rel()  

plt.show()