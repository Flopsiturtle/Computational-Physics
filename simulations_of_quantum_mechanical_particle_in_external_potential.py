# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:33:12 2024

@author: Florian Hollants
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#declaring variables
H_BAR = .000001
MASS = .00001
D = 2
N = 1000
L=1
A = L/N
OMEGA=1
R=1
T=1
M=100






MU=MASS*OMEGA*R**2/H_BAR
EPSILON = A/R
TAU=OMEGA*T/M




shape = (N,)  # tuple of D N's to shape ndarrays correctly
origin = (0,)  # tuple of D zeros to call position in grid later
for _ in range(D-1):
    shape += (N,)
    origin += (0,)

#Psi=np.arange(N**D).reshape(shape).astype('complex')

Psi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
Phi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
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


def laplace(func):
    """calculating the laplacian of ndarray"""
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap/A**2


def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*MU*EPSILON**2)*laplace(func)+potential()

#make sure to save result in regular intervals to use for animation function
def time_evol(func):
    """time evolution using second-order Fourier transform"""
    for _ in range(M):
        func = func - 1j*hamilton(func)*TAU-TAU**2*hamilton(hamilton(func))/2
    return func


fig, ax = plt.subplots()
line = ax.plot(np.linspace(-L/2, L/2, N), Psi)
ax.set(xlim=[-L/2,L/2], ylim=[-10,10])
ax.legend()

def animate(frame):
    """animates time evolution"""
    y=
    return 

ani = animation.FuncAnimation(fig=fig, func=time_evol, frames=40, interval=30)
plt.show()


#print(Psi)
print(inner_product(hamilton(Psi),Psi)-inner_product(Psi,hamilton(Psi)))
print(np.average(hamilton(2*Psi+Phi)-2*hamilton(Psi)-hamilton(Phi)))
#print(laplace(Psi))
#print(hamilton(Psi))
#print(time_evol(Psi))
