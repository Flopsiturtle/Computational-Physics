#Florian Telleis
import numpy as np
# import cmath
import math
import matplotlib.pyplot as plt


""" --- constans and parameters --- """

# D = len(wave.shape)
# N = (wave.shape)[0]
A = 0.3  # for viewing good 0.13             # assumption: input is a | can also be variant of L=N*a with N=(wave.shape)[0]  \\ [0] is arbitrary because quadratic matrix


R = 3  # length from zero-point to potential-valleys 
M = 1   # mass of point particle
W = 1   # ... in units of time^-1
H_BAR =1#!!! actually: 6.62607015*10**(-34)    # J*s


mu = (M*W*R**2)/H_BAR
epsilon = A/R




""" --- hamiltonian and CO. --- """


def wave_to_lattice(wave):   # transformation can also be done implicitly in a later function 
    """transform given wavefunction "wave" to lattice units (discrete)"""
    D = len(wave.shape)             
    return A**(D/2)*wave        

# phi = wave_to_lattice(wave,a)


def laplace_in_lattice(phi):    # input phi has to be in lattice units
    """calculate the laplacian of a wavefunction "phi" """
    shape = phi.shape
    D = len(shape)      # assumption: quadratic arrays; for example 2D: NxN
    laplace = np.zeros(shape)
    for i in np.arange(0,D):
        laplace = laplace + np.roll(phi,1,i) - 2*phi + np.roll(phi,-1,i)
    return laplace


def potential_in_lattice(phi):
    """define the potential in lattice units with the same dimensions as input wavefunction "phi" """
    V = np.zeros(phi.shape)
    N = phi.shape[0]
    for index, value in np.ndenumerate(phi):
        index_arr = np.array(index)
        V[index] = mu/8 * (epsilon**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2       
    return V        
                                # move potential to lattice centrum to obtain all wanted values! -> index_arr-int(n/2)
                                # i understood the lattice moving for the potential
                            ##### BUT how does this influence in which way we input our given wavefunction??? is it already central????

def hamiltonian_in_lattice(phi):
    """calculate the hamiltonian of a given wavefunction "phi" """    
    #phi = wave_to_lattice(wave) # phi already input
    H = - 1/(2*mu*epsilon**2)*laplace_in_lattice(phi) + potential_in_lattice(phi) * phi
    return H



# scalar product
def scal_prod(phi1,phi2):
    """calculates the scalar product/inner product of two wavefunctions"""
    return np.sum(np.multiply(np.conjugate(phi1),phi2))





"""check for characteristics: - linear, - hermitian, -""" 
#!#!#!#!#! checks were done before upload of scipt with the info

# wrong
def check_hermitian(hamiltonian,error):      # geht nur für 2D arrays durch np.matrix!!!
    ham_matrix = np.matrix(hamiltonian)
    ham_adj = ham_matrix.getH()
        ## dont check hermitian of hamiltonian on phi but hamiltonian itself!
    if abs((ham_matrix - ham_adj)).all() <= error:
        result = "hermitian"
    else:
        result = "non-hermitian"
    return print(result)
# wrong

def check_linear(wave,error):
                # hamilton(lambda*phi) == lambda*hamilton(phi)      # -> welche Zahl als Beispiel?: einfach mal 27.9
                # hamilton(phi+phi´) == hamilton(phi) + hamilton(phi´)      # wie phi´ aus phi erzeugen?: wave@wave
    wave2 = wave@wave
    linear1 = hamiltonian_in_lattice(27.9*(wave-wave2))
    linear2 = 27.9*(hamiltonian_in_lattice(wave)-hamiltonian_in_lattice(wave2))
    if (abs(linear1 == linear2)).all() <= error:
        result = "linear"
    else:
        result = "non-linear"
    return print(result)






""" --- integrators --- """

M = 2000     # large value
T = 2   # time
tau = T/M   # time step


def so_integrator(phi0):
    """solves the time dependent schrödinger equation for a given wavefunction "phi0" with the second-order integrator"""
    start = phi0
    for m in np.arange(1,M+1):
        iteration = start - 1j*tau*hamiltonian_in_lattice(start) - 1/2*tau**2*hamiltonian_in_lattice(hamiltonian_in_lattice(start))
        start = iteration
    return iteration
    # I think this is correctly implemented
    ##### BUT had a question here: how do we show complex numbers in animation????



""" --- check unitarity --- """

def check_so_unitarity(phi0,error):   
    """checks the unitarity of the second-order evolution operator via conservation of the norm of states"""   
    phitau = so_integrator(phi0)
    if abs((scal_prod(phi0,phi0).real)-(scal_prod(phitau,phitau))) <= error:
        result = "OK, evolution operator is unitary."
    else:
        result = "WARNING, evolution operator is not unitary."
    return print(result)



""" --- check error --- """

# wrong
def exp_array(array,k):
    array = np.identity() + array
    for i in np.arange(0,k+1):
        array = (array**k)/math.factorial(k)    ### array**k is different than wanted definition 
    return array
# wrong








# ----- tests -----


test_2D = np.array([[[1.3,4.27],[3.6,5.9]]])
test_2D2 = np.array([[[2,5+1j],[3,9+4j]]])
test_3D = np.array([[[2,32],[2,1]],[[2,7],[2,3]]])


size = 30
test_2D3 = np.random.rand(size,size)+ 1j * np.random.rand(size,size)

f = potential_in_lattice(test_2D3)
g = hamiltonian_in_lattice(test_2D3).real
h = abs(so_integrator(test_2D3))

plt.imshow(f, interpolation='none')
plt.title('potential: A={0}, R={1}, size={2}'.format(A,R,size))
plt.show()
plt.imshow(abs(test_2D3), interpolation='none')
plt.title('abs-wavefunction: A={0}, R={1}, size={2}'.format(A,R,size))
plt.show()
plt.imshow(h, interpolation='none')
plt.title('abs-so_integrator: A={0}, R={1}, size={2}, M={3}, T={4}'.format(A,R,size,M,T))
plt.show()





#print(laplace_in_lattice(test_2D3))
#check_hermitian(test_2D3,1)
#check_linear(test_2D3,0.01)


#print(so_integrator(test_2D3))
#a=so_integrator(test_2D3)

#check_so_unitarity(test_2D3,0.001)






#print(test_2D3)


#print(exp_array(test_2D,5))

#print(round(scal_prod(test_2D,test_2D).real,7))

#print(round(scal_prod(so_integrator(test_2D),so_integrator(test_2D)).real,8))


#print(check_hermitian(test_3D)) # geht nicht durch 3D

#print(check_hermitian(test_2D))
#print(check_hermitian(test_2D2))

#print(check_linear(test_2D,......))        inputs fehlen für hermitian


#print(np.arange(1,10+1))


#print(test)
#print(np.roll(test,1,0))

# axis 2 -> x
# axis 1 -> y
# axis 0 -> z




#input()