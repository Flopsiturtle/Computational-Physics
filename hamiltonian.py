#Florian Telleis
import numpy as np

def wave_to_lattice(wave,a):   # transform given wavefunction to lattice units
    D = len(wave.shape)             #  \\\-> transformation can also be done implicitly in a later function 
    return a**(D/2)*wave

# phi = wave_to_lattice(wave,a)    # execute for given wavefunction



def laplace_in_lattice(phi):    # input phi has to be in lattice units
    shape = phi.shape
    D = len(shape)      # assumption: quadratic arrays; for example 2D: NxN
    laplace = np.zeros(shape)
    for i in np.arange(0,D):
        laplace += np.roll(phi,1,i) - 2*phi + np.roll(phi,-1,i)
    return laplace


def potential_in_lattice(mu,epsilon,n):
    return mu/8 * (epsilon**2*n**2-1)**2    # ??? whats with n vector?????



def hamiltonian_in_lattice(wave,a,mu,epsilon,n):    # input phi has to be in lattice units, else phi=a**(D/2)*wave
    phi = wave_to_lattice(wave,a)
    H = - 1/(2*mu*epsilon**2)*laplace_in_lattice(phi) + potential_in_lattice(mu,epsilon,n) * phi
    return H





test_3D=np.array([[[2,32],[2,1]],[[2,7],[2,3]]])

print(laplace_in_lattice(test_3D))



#print(np.arange(0,2))


#print(test)
#print(np.roll(test,1,0))

# axis 2 -> x
# axis 1 -> y
# axis 0 -> z
