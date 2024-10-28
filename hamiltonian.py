#Florian Telleis
import numpy as np
import cmath


def wave_to_lattice(wave,a):   # transform given wavefunction to lattice units  \\\-> transformation can also be done implicitly in a later function 
    D = len(wave.shape)             
    return a**(D/2)*wave        # assumption: input is a | can also be variant of L=N*a with N=(wave.shape)[0]  \\ [0] is arbitrary because quadratic matrix

# phi = wave_to_lattice(wave,a)


def laplace_in_lattice(phi):    # input phi has to be in lattice units
    shape = phi.shape
    D = len(shape)      # assumption: quadratic arrays; for example 2D: NxN
    laplace = np.zeros(shape)
    for i in np.arange(0,D):
        laplace += np.roll(phi,1,i) - 2*phi + np.roll(phi,-1,i)
    return laplace


def potential_in_lattice(mu,epsilon,phi):
    D = len(phi.shape)
    n = np.zeros(D)
#    for i in np.arange(D):                     # ??? whats with n vector?????
#        ....
    V = mu/8 * (epsilon**2*n@n-1)**2       # you could create array of all the n@n and than np.multiply with phi
    return V            



def hamiltonian_in_lattice(wave,a,mu,epsilon,n):    # phi has to be in lattice units, else phi=a**(D/2)*wave
    phi = wave_to_lattice(wave,a)
    H = - 1/(2*mu*epsilon**2)*laplace_in_lattice(phi) + potential_in_lattice(mu,epsilon,n) * phi
    return H



# check for characteristics: - linear, - hermitian, - 

def check_hermitian(hamilton):      # geht nur für 2D arrays durch np.matrix!!!
    ham_matrix = np.matrix(hamilton)
    ham_adj = ham_matrix.getH()
    ham_adj_matrix = np.matrix(ham_adj)
    if (ham_matrix == ham_adj_matrix).all():
        a = "hermitian"
    else:
        a = "non-hermitian"
    return a

def check_linear(hamilton):
    # hamilton(lambda*phi) == lambda*hamilton(phi)      # -> welche Zahl als Beispiel?
    # hamilton(phi+phi´) == hamilton(phi) + hamilton(phi´)      # wie phi´ aus phi erzeugen?



# ----- tests -----


test_2D = np.array([[[1,1],[1,1]]])
test_2D2 = np.array([[[2,5+1j],[3,9+4j]]])
test_3D = np.array([[[2,32],[2,1]],[[2,7],[2,3]]])


print(laplace_in_lattice(test_3D))

#print(check_hermitian(test_3D)) # geht nicht durch 3D

print(check_hermitian(test_2D))
print(check_hermitian(test_2D2))






#print(np.arange(0,2))


#print(test)
#print(np.roll(test,1,0))

# axis 2 -> x
# axis 1 -> y
# axis 0 -> z
