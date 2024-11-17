# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:33:12 2024

@author: Florian Hollants

Goal: Animate Wavefunction that starts as gaussian centered at one minimum of the potential
"""
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import stats 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#declaring variables
N = 200 # number of points in each direction of D-dimensional lattice  # if wavefunction is given: N = (wave.shape)[0]    # \\ [0] is arbitrary because quadratic matrix
A = 0.3  # spacing between lattice points   # assumption: A is input | can also be variant of L=N*a  
"""potential/hamiltonian"""
R = 18  # length from zero-point to potential-valleys 
M = 0.1   # mass of point particle
W = 5   # frequency
H_BAR = 1#!!! actually: 6.62607015*10**(-34)    # J*s
mu = (M*W*R**2)/H_BAR
epsilon = A/R
D=1
M = 10000  # large value
T = 10      # time
tau = T/M   # time step

FRAMES = 200    # number of frames in final animation
FPS = int(FRAMES/T)     # number of frames per second if given time T is real time in seconds
#FPS = 23


shape = (N,)  # tuple of D N's to shape ndarrays correctly
origin = (0,)  # tuple of D zeros to call position in grid later
for _ in range(D-1):
    shape += (N,)
    origin += (0,)


def gaussian_1D(mean,sigma): 
    x_data = np.arange(0, N) 
    y_data = stats.norm.pdf(x_data, mean, sigma) 
    return y_data 


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
    """calculating the free_hamiltonian"""
    return -1/(2*mu*epsilon**2)*laplace(func)



def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*mu*epsilon**2)*laplace(func)+np.multiply(V,func)

def test_linearity(Hamiltonian,psi_in,iterations):
    shape = np.shape(psi_in)
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = Hamiltonian(alpha[i]*psi1 + beta[i]*psi2)
        RHS = alpha[i]*Hamiltonian(psi1) + beta[i]*Hamiltonian(psi2)
        error = np.max(np.abs(LHS - RHS))
        print(error)


def test_hermiticity(Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = np.sum(np.multiply(np.conjugate(psi1), Hamiltonian(psi2)))
        RHS = np.sum(np.multiply(np.conjugate(Hamiltonian(psi1)), psi2))
        print(np.abs(LHS - RHS))

def test_positivity(Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        print("Potential:"  ,np.sign(np.sum(np.multiply(np.conjugate(psi1), potential(psi1))).real))
        print("Hamiltonian:" , np.sign(np.sum(np.multiply(np.conjugate(psi1), Hamiltonian(psi1))).real))

def test_eigenvectors(Kinetic_Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    D = len(shape)
    plane_wave = np.zeros(shape, dtype=complex)
    for i in range(iterations):
        k = np.random.randint(-N,N, size= D)
        eigenvalue = 0
        for index, value in np.ndenumerate(psi_in):
            plane_wave[index] = np.exp(2*np.pi * 1j * np.dot(np.array(index),k)/N)
        LHS = Kinetic_Hamiltonian(plane_wave)
        for i in range(len(k)):
            eigenvalue += (np.sin(np.pi/N * k[i]))**2
        RHS = 2/(mu*epsilon**2)*eigenvalue * plane_wave
        print(np.max(np.abs(LHS - RHS )))




def so_integrator(func,M):
    """solves the time dependent schrödinger equation for a given wavefunction "phi0" with the second-order integrator"""
    start = func
    for m in np.arange(0,M):
        iteration = start - 1j*tau*hamilton(start) - 1/2*tau**2*hamilton(hamilton(start))
        start = iteration
    return iteration




def images(func, integ):
    """ creates a list of the calculated wavefunctions for all timesteps of the second-order time evolution """
    start = func
    ims = []
    ims.append(start)
    for m in np.arange(0,M):
        iteration = integ(start,1)
        start = iteration
        ims.append(iteration)
    return ims

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



def Strang_Splitting(psi_in,M):
    for i in range(M):
        psi_out = Exponential_potential(psi_in, tau)
        psi_out = fft(psi_out)
        psi_out = Exponential_kinetic(psi_out, tau)
        psi_out = ifft(psi_out)
        psi_out = Exponential_potential(psi_out,tau)
        psi_in = psi_out
    return psi_out


def test_unitarity(psi_in, iterations):
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = phi/np.sqrt(np.dot(np.conjugate(phi),phi))
    for i in range(iterations):
        wave = Strang_Splitting(phi,1)
        norm = np.sqrt(np.dot(np.conjugate(wave), wave))
        print(np.abs(1-norm))
        phi = wave

def test_Strang_linearity(psi_in, iterations):
    shape = np.shape(psi_in)
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = Strang_Splitting((alpha[i]*psi1 + beta[i]*psi2), 1)
        RHS = alpha[i]*Strang_Splitting(psi1, 1) + beta[i]*Strang_Splitting(psi2,1)
        error = np.sum(np.abs(LHS - RHS))
        print(error)
       
def test_energy_conserv(psi_in, iterations):
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = phi/np.sqrt(np.dot(np.conjugate(phi),phi))
    energy0 = np.dot(np.conjugate(phi), hamilton(phi))
    for i in range(iterations):
        wave = Strang_Splitting(phi, 1)
        energy0 = np.dot(np.conjugate(phi),hamilton(phi))
        energy1 = np.dot(np.conjugate(wave),hamilton(wave))
        error = np.abs(energy1 - energy0)
        print(error)
        phi = wave        
      






#fig = plt.figure()
#axis = plt.axes(xlim=(0,N-1),ylim =(0, 0.002))  # for constant axis

#fig, axis = plt.subplots()     # for variable axis

fig, (ax1, ax2) = plt.subplots(1,2)
line1, = ax1.plot([], [])  
line2, = ax2.plot([], [])  
ax1.set_xlim(0,N-1)
ax1.set_ylim(0,0.4)
ax2.set_xlim(0,N-1)
ax2.set_ylim(0,0.4)

def init1():  
    line1.set_data([], []) 
    return line1, 

def init2():  
    line2.set_data([], []) 
    return line2, 
   
   
def animate_so_integrator(i): 
    """function that gets called for animation"""
    """takes a number of arrays from calculated "images" corresponding to FRAMES and sets a line data"""
    global FRAMES,images_so,M
    i2 = int(i*M/(FRAMES-1)) # dont use all the frames from the time evolution
    y = abs(images_so[i2])**2
    x = np.arange(0,len(y)) 
    #axis.set_ylim(0, max(y)*1.1) # for variable axis
    line1.set_data(x, y) 
    return line1, 

def animate_strang_integrator(i): 
    """function that gets called for animation"""
    """takes a number of arrays from calculated "images" corresponding to FRAMES and sets a line data"""
    global FRAMES,images_strang,M
    i2 = int(i*M/(FRAMES-1)) # dont use all the frames from the time evolution
    y = abs(images_strang[i2])**2
    x = np.arange(0,len(y)) 
    #axis.set_ylim(0, max(y)*1.1) # for variable axis
    line2.set_data(x, y) 
    return line2, 


Psi=gaussian_1D(25,10)*10
V = potential(Psi)
iterations = 10


#test_eigenvectors(kinetic_hamilton, Psi, iterations)
test_energy_conserv(Psi, iterations)



images_so = images(Psi, so_integrator)    # creates the list of arrays for our test function
images_strang = images(Psi, Strang_Splitting)



anim = animation.FuncAnimation(fig, animate_so_integrator, init_func = init1, frames = FRAMES, interval = 1000/FPS, blit = False) 
#plt.show()
anim2 = animation.FuncAnimation(fig, animate_strang_integrator, init_func = init2, frames = FRAMES, interval = 1000/FPS, blit = False) 



""" difference in last M for our tau"""
""" question is: do we run the code three times for different tau???"""

fig = plt.figure()
axis = plt.axes(xlim=(0,N-1),ylim =(0, 0.002))  # for constant axis
axis.plot(np.arange(0,200),np.abs(abs(images_so[M])**2-abs(images_strang[M])**2))
axis.set_ylim(0, max(np.abs(abs(images_so[M])**2-abs(images_strang[M])**2))*1.1) 
#plt.plot(np.arange(0,200),np.abs(abs(images_so[M])**2-abs(images_strang[M])**2))



plt.show()


































"""
fig = plt.figure()
axis = plt.axes(xlim=(0,N-1),ylim =(0, 0.002))  # for constant axis
#fig, axis = plt.subplots()     # for variable axis

line, = axis.plot([], [])  

def init():  
    line.set_data([], []) 
    return line, 
   
def animate_so_integrator(i): 
    """"function that gets called for animation""""
    """"takes a number of arrays from calculated ""images" "corresponding to FRAMES and sets a line data""""
    global FRAMES,images_so,M
    i2 = int(i*M/(FRAMES-1)) # dont use all the frames from the time evolution
    y = abs(images_so[i2])**2
    x = np.arange(0,len(y)) 
    #axis.set_ylim(0, max(y)*1.1) # for variable axis
    line.set_data(x, y) 
    return line, 

def animate_strang_integrator(i): 
    """"function that gets called for animation""""
    """"takes a number of arrays from calculated ""images"" corresponding to FRAMES and sets a line data""""
    global FRAMES,images_strang,M
    i2 = int(i*M/(FRAMES-1)) # dont use all the frames from the time evolution
    y = abs(images_strang[i2])**2
    x = np.arange(0,len(y)) 
    #axis.set_ylim(0, max(y)*1.1) # for variable axis
    line.set_data(x, y) 
    return line, 





Psi=gaussian_1D(25,10)
V = potential(Psi)


images_so = images(Psi, so_integrator)    # creates the list of arrays for our test function
images_strang = images(Psi, Strang_Splitting)



#anim = animation.FuncAnimation(fig, animate_so_integrator, init_func = init, frames = FRAMES, interval = 1000/FPS, blit = True) 
#plt.show()
anim = animation.FuncAnimation(fig, animate_strang_integrator, init_func = init, frames = FRAMES, interval = 1000/FPS, blit = True) 



plt.show()








"""