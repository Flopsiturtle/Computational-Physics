

import numpy as np

"""Constants and Definitions"""

N = 100
epsilon = 1
mu = 1
initial = np.random.rand(N,N)

"""Calculate Hamiltonian of a single particle in a double well potential."""    
                        
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




"""Hamiltonian Tests"""

def test_linearity(Hamilton_operator,psi_in,iterations, error):
    """Tests the linearity of the given Hamilton operator. An arbitrary wave function
    psi_in must be given to define the dimensions. 
    Iterations defines the number of tests to be executed with random scalars 
    and wavefunctions. Error defines the error bound on checking for linearity.
    Returns statement regarding the linearity."""
    shape = np.shape(psi_in)
    truth_values = np.array([])
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = Hamiltonian(alpha[i]*psi1 + beta[i]*psi2)
        RHS = alpha[i]*Hamiltonian(psi1) + beta[i]*Hamiltonian(psi2)
        if (np.all(np.abs(LHS - RHS) < error)):
            print('Hamiltonian is linear!')
        else:
            print('Hamiltonian is not linear!')
            
def test_hermiticity(Hamilton_operator, psi_in, iterations):
    """Tests the hermiticity of the given Hamilton operator. 
    An arbitrary wave function psi_in must be given to define the dimensions. 
    Iterations defines the number of tests to be executed with random wavefunctions. 
    Error defines the error bound on checking for hermiticity. 
    Returns statement regarding the hermiticity."""
    shape = np.shape(psi_in)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = np.sum(np.multiply(np.conjugate(psi1), Hamiltonian(psi2)))
        RHS = np.sum(np.multiply(np.conjugate(Hamiltonian(psi1)), psi2))
        print(np.sum(np.abs(LHS - RHS)))
         
        
        
def test_positivity(Hamilton_operator, potential_operator, psi_in, iterations):
    """Tests the positivity of the given Hamilton operator and Potential. 
    An arbitrary wave function psi_in must be given to define the dimensions. 
    Iterations defines the number of tests to be executed with random wavefunctions. 
    Returns statement regarding the positivity."""
    shape = np.shape(psi_in)
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        print("Potential:"  ,np.sum(np.multiply(np.conjugate(psi1), Potential(psi1))).real)
        print("Hamiltonian:" , np.sum(np.multiply(np.conjugate(psi1), Hamiltonian(psi1))).real)
       
        
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
        print(np.sum(np.abs(LHS - RHS )))

test_eigenvectors(Kinetic_Hamiltonian, initial, 1) 
test_positivity(Hamiltonian, Potential, initial, 1)
test_hermiticity(Hamiltonian, initial, 1)

            

            
     



            

            
     

