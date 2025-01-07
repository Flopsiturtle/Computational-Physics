import numpy as np
import matplotlib.pyplot as plt

import hamiltonian
import integrators
import variables
from variables import *

''' --- testing dependences on M and tau --- '''

####### feedback from him: maybe run rel() in other function because so much time so wen can run these tests seperately


def rel():
    global M, tau
    M_save = M
    tau_save = tau
    Ms = np.linspace(3000, 10000, 8).astype(int)
    E_so = []
    E_st = []
    norm_so = []
    avg_diff = []
    E_init = variables.inner_product(Psi, hamiltonian.hamilton(Psi)).real/(variables.inner_product(Psi, Psi)).real
    for m in Ms:
        M = m
        tau = 10/M
        so = integrators.so_integrator(Psi, M, tau)
        st = integrators.Strang_Splitting(Psi, M, tau)
        E_so.append(variables.inner_product(so, hamiltonian.hamilton(so)).real)
        E_st.append(variables.inner_product(st, hamiltonian.hamilton(st)).real)
        norm_so.append(variables.inner_product(so, so).real)
        avg_diff.append(np.average(np.abs(so-st)))
        print(str(M) + " out of " + str(Ms[-1]))
    M = M_save
    tau = tau_save
    
    figure, axs = plt.subplots(2,2)
    axs[0,0].plot(Ms, np.array(E_so)/np.array(norm_so)-E_init, marker = 'o', label=r'$\frac{\langle\hat{\Psi}_{so}|\hat{H}|\hat{\Psi}_{so}\rangle}{\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle}-\langle\hat{\Psi}_0|\hat{H}|\hat{\Psi}_0\rangle$')  #,   
    axs[0,1].plot(Ms, E_st-E_init, marker = 'o', label=r'$\langle\hat{\Psi}_{st}|\hat{H}|\hat{\Psi}_{st}\rangle-\langle\hat{\Psi}_0|\hat{H}|\hat{\Psi}_0\rangle$')
    axs[1,0].plot(Ms, norm_so-variables.inner_product(Psi, Psi).real, marker = 'o', label=r'$\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle-\langle\hat{\Psi}_0|\hat{\Psi}_0\rangle$')
    axs[1,1].plot(Ms, avg_diff, marker = 'o', label="avg($|\hat{\Psi}_{so}-\hat{\Psi}_{st}|$)")
    for ax in axs.flat:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('M')
        ax.legend(fontsize=28)
    
    return Ms, E_so, E_st, norm_so, avg_diff

n, Psi = variables.gaussian_1D(-int(N/4),int(N/16))
Psi = variables.normalize(Psi)

Ms, E_so, E_st, norm_so, avg_diff = rel()  

E_init = variables.inner_product(Psi, hamiltonian.hamilton(Psi)).real/(variables.inner_product(Psi, Psi)).real
slopes = [(np.log10(np.array(E_so)[-1]/np.array(norm_so)[-1]-E_init) - np.log10(np.array(E_so)[0]/np.array(norm_so)[0]-E_init))/(np.log10(Ms[-1]) - np.log10(Ms[0])),
          (np.log10(np.array(E_st)[-1]-E_init) - np.log10(np.array(E_st)[0]-E_init))/(np.log10(Ms[-1]) - np.log10(Ms[0])),
          (np.log10(np.array(norm_so)[-1]-variables.inner_product(Psi, Psi).real) - np.log10(np.array(norm_so)[0]-variables.inner_product(Psi, Psi).real))/(np.log10(Ms[-1]) - np.log10(Ms[0])),
          np.log10(np.array(avg_diff)[-1]) - np.log10(np.array(avg_diff)[0])]

print(slopes)
plt.show()



"""
# testing from above displayed without tabulars
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
"""
