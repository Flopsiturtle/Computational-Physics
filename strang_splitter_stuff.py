

"""Strang-Splitting-Integrator"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
%matplotlib qt



L = 1
N = 1000

A = L/N
r = L/4
epsilon = A/r
mass = 10
omega = 1
hbar = 1
#mu = mass*omega*r**2/hbar
mu = 1
centre = 800
sigma = 2
grid = np.array([i for i in range(N)])

def discrete_norm(psi_in):
    return np.multiply(np.conjugate(psi_in), psi_in)
k = 1

initial = np.exp(-(grid-centre)**2/(2*sigma**2) + 1j*grid*k)
initial = initial/np.sqrt(np.dot(np.conjugate(initial),initial))
#initial = 30*np.exp(2*np.pi*1j*grid*k/N)
#initial = 1/N*initial
#print(initial)
def Potential_values(psi_in):
    """Calculates double-well potential(consider the grid!) and 
    applies it to a given wavefunction psi_in. Returns wavefunction of same shape. """
    shape = np.shape(psi_in)
    squares = np.zeros(shape)
    for index, value in np.ndenumerate(psi_in):
        squares[index] = np.dot(np.array(index)-int(N/2) , np.array(index)-int(N/2))
    potential_term = mu/8 * (epsilon**2 * squares - 1)**2
    return potential_term

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

def Strang_Splitting(psi_in, tau):
    psi_out = Exponential_potential(psi_in, tau)
    psi_out = sp.fft.fftn(psi_out)
    psi_out = Exponential_kinetic(psi_out, tau)
    psi_out = sp.fft.ifftn(psi_out)
    psi_out = Exponential_potential(psi_out,tau)
    return psi_out
    
def test_unitarity(psi_in, tau, step_number):
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = phi/np.sqrt(np.dot(np.conjugate(phi),phi))
    for i in range(step_number):
        wave = Strang_Splitting(phi, tau)
        norm = np.sqrt(np.dot(np.conjugate(wave), wave))
        print(np.abs(1-norm))
        phi = wave

def test_Strang_linearity(psi_in, tau, iterations):
    shape = np.shape(psi_in)
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = Strang_Splitting((alpha[i]*psi1 + beta[i]*psi2), tau)
        RHS = alpha[i]*Strang_Splitting(psi1, tau) + beta[i]*Strang_Splitting(psi2,tau)
        error = np.sum(np.abs(LHS - RHS))
        print(error)

def Laplacian(psi_in):
    """Calculates discrete Laplacian applied to a given Wavefunction psi_in. 
    Returns wavefunction of same shape."""
    D = len(np.shape(psi_in))
    psi_out = -2*D*psi_in
    for i in range(D):
        psi_out += np.roll(psi_in,1, axis = i) + np.roll(psi_in, -1, axis = i)
    return psi_out

def Potential(psi_in):
    """Calculates double-well potential(consider the grid!) and 
    applies it to a given wavefunction psi_in. Returns wavefunction of same shape. """
    shape = np.shape(psi_in)
    squares = np.zeros(shape)
    for index, value in np.ndenumerate(psi_in):
        squares[index] = np.dot(np.array(index)-int(N/2) , np.array(index)-int(N/2))
    potential_term = mu/8 * (epsilon**2 * squares - 1)**2
    psi_out = np.multiply(potential_term, psi_in)
    return psi_out

def Kinetic_Hamiltonian(psi_in):
    psi_out = -1/(2*mu*epsilon**2)*Laplacian(psi_in)
    return psi_out
        
def Hamiltonian(psi_in):
    """Calculates Hamiltonian applied to a wavefunction psi_in. 
    Returns wavefunction of same shape."""
    psi_out = -1/(2*mu*epsilon**2)*Laplacian(psi_in) + Potential(psi_in)
    return psi_out
       
def test_energy_conserv(psi_in, tau, step_number):
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = phi/np.sqrt(np.dot(np.conjugate(phi),phi))
    energy0 = np.dot(np.conjugate(phi), Hamiltonian(phi))
    print(energy0)
    for i in range(step_number):
        wave = Strang_Splitting(phi, tau)
        energy = np.dot(np.conjugate(wave),Hamiltonian(wave))
        print(energy)
        phi = wave        

fig, ax = plt.subplots()
ax.set_xlim(600,N)
ax.set_ylim(-0.1,0.2)
line2 = ax.plot(grid, discrete_norm(initial))[0]
plt.plot(grid, Potential_values(initial))
plt.plot(grid, discrete_norm(initial))
running = initial
def update(i):
    global running
    running = Strang_Splitting(running, 0.0001)
    line2.set_xdata(grid)
    line2.set_ydata(discrete_norm(running))
    return line2
animation = FuncAnimation(fig, func = update, frames = 500,interval = 100)
plt.show() 

        
        
    

   




    
    
    
    

    
    
    
    
    
    
    
    
    
    
    


    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
