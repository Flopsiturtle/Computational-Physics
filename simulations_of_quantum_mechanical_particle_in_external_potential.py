# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:33:12 2024

@author: Florian Hollants
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.constants as const
#declaring variables
H_BAR = .0001 #const.hbar*10**5
MASS = .0001 #const.electron_mass*10**5
D = 1
N = 100
L = 1#N*const.physical_constants['atomic unit of length'][0]*10**5
A = L/N#const.physical_constants['atomic unit of length'][0]*10**5
OMEGA=1#/const.physical_constants['atomic unit of time'][0]*10**5
R=L/20
T=1#3*const.physical_constants['atomic unit of time'][0]*10**5
M=1000



MU=MASS*OMEGA*R**2/H_BAR
EPSILON = A/R
TAU=OMEGA*T/M




shape = (N,)  # tuple of D N's to shape ndarrays correctly
origin = (0,)  # tuple of D zeros to call position in grid later
for _ in range(D-1):
    shape += (N,)
    origin += (0,)

Psi=np.arange(N**D).reshape(shape).astype('complex')/N**D

#Psi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
#Phi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
#shifting the origin to the center of the grid


V=np.empty_like(Psi)
for n, _ in np.ndenumerate(Psi):
    V[n]=np.dot(np.array(n)-int(N/2),np.array(n)-int(N/2))


def potential():
    """defines the potential"""
    return MU/8*(EPSILON**2*V-1)**2


def inner_product(func1, func2):
    """calculates the inner product of two arrays"""
    return np.sum(np.multiply(np.conjugate(func1),func2))


Psi *= 1/np.sqrt(inner_product(Psi,Psi))


def laplace(func):
    """calculating the laplacian of ndarray"""
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap


def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*MU*EPSILON**2)*laplace(func)+potential()

#make sure to save result in regular intervals to use for animation function
def time_evol(func, n):
    """time evolution using second-order Fourier transform"""
    for _ in range(n):
        func = func - 1j*hamilton(func)*TAU-TAU**2*hamilton(hamilton(func))/2
    return func


fig, ax = plt.subplots()
line = ax.plot(np.linspace(-L/2, L/2, N), Psi.imag)[0]
ax.set(xlim=[-L/2,L/2], ylim=[-10*10**5*10**5,10*10**5*10**5])

def animate(frame):
    """animates time evolution"""
    line.set_ydata(time_evol(Psi, n=frame).imag)
    return line

#ani = animation.FuncAnimation(fig=fig, func=animate, frames=50, interval=1000)
#plt.show()


#print(Psi)
#(inner_product(hamilton(Psi),Psi)-inner_product(Psi,hamilton(Psi)))
#print(np.average(hamilton(2*Psi+Phi)-2*hamilton(Psi)-hamilton(Phi)))
#print(laplace(Psi))
#print(hamilton(Psi))
print(time_evol(Psi, 20))
print(inner_product(time_evol(Psi, 20), time_evol(Psi, 20)))