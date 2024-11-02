#Florian Telleis
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt



##### am Anfang die Konstante definieren!!!  -> nicht jedes mal in Funktion epsilon, etc!!!

# D = len(wave.shape)
# N = (wave.shape)[0]
a = 1               # assumption: input is a | can also be variant of L=N*a with N=(wave.shape)[0]  \\ [0] is arbitrary because quadratic matrix
mu = 1
epsilon = 1



def wave_to_lattice(wave):   # transform given wavefunction to lattice units  \\\-> transformation can also be done implicitly in a later function 
    D = len(wave.shape)             
    return a**(D/2)*wave        

# phi = wave_to_lattice(wave,a)


def laplace_in_lattice(phi):    # input phi has to be in lattice units
    shape = phi.shape
    D = len(shape)      # assumption: quadratic arrays; for example 2D: NxN
    laplace = np.zeros(shape)
    for i in np.arange(0,D):
        laplace = laplace + np.roll(phi,1,i) - 2*phi + np.roll(phi,-1,i)
    return laplace


def potential_in_lattice(phi):
    V = np.zeros(phi.shape)
    N = phi.shape[0]
    for index, value in np.ndenumerate(phi):
        index_arr = np.array(index)
        V[index] = mu/8 * (epsilon**2*np.dot(index_arr-N/2,index_arr-N/2)-1)**2       # this array serves as all the possible elements of V for all the n -> to multiply with all elements of phi: np.multiply(V,phi)    
    return V        
                                # move potential to lattice centrum to obtain all wanted values! -> index_arr-int(n/2)
                            # i understood the lattice moving for the potential
                            # BUT how does this influence in which way we input our given wavefunction??? is it already central????

def hamiltonian_in_lattice(phi):    
    #phi = wave_to_lattice(wave) # phi already input
    H = - 1/(2*mu*epsilon**2)*laplace_in_lattice(phi) + potential_in_lattice(phi) * phi
    return H



# scalar product
def scal_prod(phi1,phi2):
    return np.sum(np.multiply(np.conjugate(phi1),phi2))





# check for characteristics: - linear, - hermitian, - 
#!#!#!#!#! checks were done before upload of scipt with the info


# wrong
def check_hermitian(hamiltonian):      # geht nur für 2D arrays durch np.matrix!!!
    ham_matrix = np.matrix(hamiltonian)
    ham_adj = ham_matrix.getH()
        ### dont check hermitian of hamiltonian on phi but hamiltonian itself!
    if (ham_matrix == ham_adj).all():
        result = "hermitian"
    else:
        result = "non-hermitian"
    return print(result)
# wrong


def check_linear(wave):
                # hamilton(lambda*phi) == lambda*hamilton(phi)      # -> welche Zahl als Beispiel?: einfach mal 27.9
                # hamilton(phi+phi´) == hamilton(phi) + hamilton(phi´)      # wie phi´ aus phi erzeugen?: wave@wave
    wave2 = wave@wave
    A = hamiltonian_in_lattice(27.9*(wave-wave2))
    B = 27.9*(hamiltonian_in_lattice(wave)-hamiltonian_in_lattice(wave2))
    if (A == B).all():
        result = "linear"
    else:
        result = "non-linear"
    return print(result)







# integrators

M = 100
tau = 0.001


def so_integrator(phi0):
    a = phi0
    for m in np.arange(1,M+1):
        b = a - (0+1j)*tau*hamiltonian_in_lattice(a) - 1/2*tau**2*hamiltonian_in_lattice(hamiltonian_in_lattice(a))
        a = b
    return b

    # should be correctly implemented
        # BUT idea here: how do we show complex numbers in animation????


# check unitarity

def check_so_unitarity(phi0,round_stop):      
    phitau = so_integrator(phi0)
    if round(scal_prod(phi0,phi0).real,round_stop) == round(scal_prod(phitau,phitau).real,round_stop):
        result = "evolution operator is unitary"
    else:
        result = "not unitary"
    return print(result)



# check error

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




print(laplace_in_lattice(test_2D))

print(so_integrator(test_2D))

check_so_unitarity(test_2D,5)




print(test_2D**0)


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
