# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:33:12 2024

@author: Florian Hollants
"""

import numpy as np
import scipy.integrate
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation


L = 60
N = 100 # number of points in each direction of D-dimensional lattice  # if wavefunction is given: N = (wave.shape)[0]    # \\ [0] is arbitrary because quadratic matrix
A = L/N  # spacing between lattice points   # assumption: A is input | can also be variant of L=N*a  

R = 18  # length from zero-point to potential-valleys 
Mass = 0.475   # mass of point particle
W = 1   # frequency
H_BAR = 1  #!!! actually: 6.62607015*10**(-34)    # J*s

mu = (Mass*W*R**2)/H_BAR
epsilon = A/R

def hamiltonian(N, dx, V=None):
    L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0 , 1], shape=(N,N))
    H = -1/(2*mu*epsilon**2)*L#/(2*dx**2)
    if V is not None:
        H += scipy.sparse.spdiags(V, 0, N, N)
    return H.tocsc()

def time_evolution_operator(H, dt):
    U = scipy.sparse.linalg.expm(-1j*H*dt).toarray()
    U[(U.real**2 + U.imag**2)<1E-10] = 0
    return scipy.sparse.csc_matrix(U)

def simulate(psi, H, dt):
    U = time_evolution_operator(H, dt)
    t = 0
    while True:
        yield psi, t*dt
        psi = U @ psi
        t += 1

def riemann(arr, dx):
    return np.sum(np.array(arr)*dx)
    
def expec(arr, mat = None):
    if mat is None:
        return np.dot(arr, arr)
    return(np.dot(arr, mat @ arr))


def probability_density(psi):
    return psi.real**2+psi.imag**2

def gaussian_wavepacket(x, x0, sigma0, p0):
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A*np.exp(1j*p0*x -((x-x0)/(2*sigma0))**2)


print(np.max(np.linalg.eig(hamiltonian(100,60/100).toarray())[0]))


"""
N = 2000
x, dx = np.linspace(-100, 100, N, endpoint=False, retstep=True)

r=35
V = (x**2-r**2)**2/15000   #V = (x/32)**2     #
psi0 = gaussian_wavepacket(x, x0=-35, sigma0=3.0, p0=15.4)
H = hamiltonian(N, dx, V = V)

T = [0]
E = [riemann(expec(psi0, H), dx)]
sim = simulate(psi0, H, dt=.1)


def update(frames):
    y, t = next(sim)
    line.set_ydata(probability_density(y))
    T.append(t)
    E.append(np.abs(riemann(expec(y, H), dx)))
    line3.set_ydata(E)
    line3.set_xdata(T)
    ax[1].set(xlim=[0, t], ylim=[0, np.max(E)])
    return line

fig, ax = plt.subplots(1, 2)

ax[1].set(ylim=[0, E[0]])
line = ax[0].plot(x, probability_density(psi0))[0]
line2 = ax[0].plot(x, V/50000)[0]
line3 = ax[1].plot(T, E)[0]

ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=50)
plt.show()
"""