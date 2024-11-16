# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:21:31 2024

@author: flori
"""

import numpy as np
from scipy.fftpack import fft, ifft
from scipy import stats 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

D=1 #dimension
M = 1000  # large value
T = 1  #time
L = 60  #length
N = 200 # number of points in each direction of D-dimensional lattice  # if wavefunction is given: N = (wave.shape)[0]    # \\ [0] is arbitrary because quadratic matrix
A = L/N  # spacing between lattice points   # assumption: A is input | can also be variant of L=N*a  

R = L*3/10  # length from zero-point to potential-valleys 
Mass = .475   # mass of point particle
W = 1 # frequency
H_BAR = 1#!!! actually: 6.62607015*10**(-34)    # J*s

mu = (Mass*W*R**2)/H_BAR
epsilon = A/R
tau = W*T/M   # time step


FRAMES = 200    # number of frames in final animation
FPS = int(FRAMES/T)     # number of frames per second if given time T is real time in seconds
#FPS = 23




def gaussian_1D(mean,sigma): 
    x_data = np.arange(-int(N/2), int(N/2)) 
    y_data = stats.norm.pdf(x_data, mean, sigma)*np.exp(-5j*x_data) 
    return x_data, y_data 

def inner_product(func1, func2):
    """calculates the inner product of two arrays"""
    return np.sum(np.multiply(np.conjugate(func1),func2))


def normalize(func):
    """normalizes input function"""
    return func*1/np.sqrt(inner_product(func,func))

def potential(func):
    """defines the potential"""
    V = np.zeros(func.shape)
    for n, _ in np.ndenumerate(func):
        index_arr = np.array(n)
        V[n]=mu/8*(epsilon**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2
    return V

def laplace(func):
    """calculating the laplacian of ndarray"""
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap


def kinetic_hamilton(func):
    """calculating the free hamiltonian"""
    return -1/(2*mu*epsilon**2)*laplace(func)


def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*mu*epsilon**2)*laplace(func)+np.multiply(V,func)


def so_integrator(func,M):
    """solves the time dependent schrödinger equation for a given wavefunction with the second-order integrator"""
    start = func
    for m in np.arange(0,M):
        iteration = start - 1j*tau*hamilton(start) - 1/2*tau**2*hamilton(hamilton(start))
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

def st_integrator(psi_in,M):
    for i in range(M):
        psi_out = Exponential_potential(psi_in, tau)
        psi_out = fft(psi_out)
        psi_out = Exponential_kinetic(psi_out, tau)
        psi_out = ifft(psi_out)
        psi_out = Exponential_potential(psi_out,tau)
        psi_in = psi_out
    return psi_out


def rel():
    global M, tau
    M_save = M
    tau_save = tau
    taus = []
    E_so = []
    E_st = []
    norm_so = []
    avg_diff = []
    for m in range(100)[1:]:
        M = 10*m
        tau = W*T/M
        so = so_integrator(Psi, M)
        st = st_integrator(Psi, M)
        taus.append(tau)
        E_so.append(inner_product(so, hamilton(so)))
        E_st.append(inner_product(st, hamilton(st)))
        norm_so.append(inner_product(so, so))
        avg_diff.append(np.average(np.abs(so-st)))
        print(M)
    M = M_save
    tau = tau_save
    
    figure, axs = plt.subplots(2,2)
    axs[0,0].plot(taus, np.array(E_so), label="E_so")#/np.array(norm_so)
    axs[0,0].plot(taus, E_st, label="E_st")
    axs[0,1].plot(taus, E_st, label="E_st")
    axs[1,0].plot(taus, norm_so, label="norm")
    axs[1,1].plot(taus, avg_diff, label="avg_diff")
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    plt.show()
    return taus, E_so, E_st, norm_so, avg_diff


n, Psi=gaussian_1D(-int(N/4),int(N/32))
Psi = normalize(Psi)
V = potential(Psi)


#taus, E_so, E_st, norm_so, avg_diff = rel()  #




"""
Psit_so = so_integrator(Psi, M)
Psit_st = st_integrator(Psi, M)
plt.plot(n*epsilon, Psi/(np.sqrt(epsilon)))
plt.plot(n*epsilon, V)
plt.plot(n*epsilon, Psit_so/(np.sqrt(epsilon)))
plt.plot(n*epsilon, Psit_st/(np.sqrt(epsilon)))
plt.show()
"""
fig, ax = plt.subplots(2,2)

line = ax[0, 0].plot(n, np.abs(Psi)**2)[0]
line2 = ax[0, 1].plot(n,np.abs(Psi)**2)[0]
line3 = ax[1, 0].plot(n, np.abs(Psi)**2*0)[0]
line4 = ax[1, 1].plot(0,inner_product(Psi, hamilton(Psi)))[0]


t = 0
running1 = Psi
running2 = Psi
running3 = Psi*0
running_t = [0]
running_E_so = [inner_product(Psi, hamilton(Psi))]
def update(frame):
    global running1, running2, running3, running_t, running_E_so, t
    t += 1
    running1 = so_integrator(running1, 10)
    running2 = st_integrator(running2, 10)
    running3 = np.abs(running1-running2)
    running_t.append(t*tau)
    running_E_so.append(np.array(inner_product(running1,hamilton(running1)))/np.array(inner_product(running1, running1)))
    
    line.set_ydata(np.abs(running1)**2)
    line2.set_ydata(np.abs(running2)**2)
    line3.set_ydata(np.abs(running1-running2)**2)
    line4.set_xdata(running_t)
    line4.set_ydata(running_E_so)
    ax[1,1].set(xlim=[0, 1.1*running_t[-1]], ylim=[0.9*running_E_so[0], 1.1*running_E_so[-1]])
    return (line)

ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=50)
plt.show()






















