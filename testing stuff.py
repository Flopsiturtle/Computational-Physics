import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator

aa = np.array([[0,1,2],[5,6,7],[11,12,13]])
print(aa)
shape_a = aa.shape

print((aa.shape)[0])

print(np.zeros(len(shape_a)))

print(aa[1,1])


bb = np.array([[[0,1,2],[5,6,7],[11,12,13]],[[0,1,2],[5,6,7],[11,12,13]],[[0,1,2],[5,6,7],[11,12,13]]])

print(np.multiply(aa,bb))

print(np.arange(2))

D = len(aa.shape)
n = np.zeros(D)
n_vector = np.zeros(aa.shape)

#for i in np.arange(D):
#    for j in np.arange((aa.shape)[i]):
#        n_vector[j] = 
#print(n_vector)

print(np.mgrid[:5,:5].transpose(1,2,0))

#print(bb@bb)


for index, value in np.ndenumerate(bb):
    print(index, value)



n_scal_prod = np.zeros(aa.shape)
for index, value in np.ndenumerate(aa):
    index_arr = np.array(index)
    n_scal_prod[index] = index_arr@index_arr
print("hier",n_scal_prod)



aa_scipy = aslinearoperator(aa)

print(aa_scipy.A)

print(LinearOperator.adjoint(aa_scipy).A)

#bb_scipy = aslinearoperator(bb)   # not ndim <= 2 !!!

#print(bb_scipy.A)

#print(LinearOperator.adjoint(bb_scipy).A)


n = np.zeros((bb.shape))
print(n)


print(np.array([0,1,2])-4)


size = 200 # is equal to N right?

gaussian_1D = 1

#test_1D = np.random.rand(size,1)+ 1j * np.random.rand(size,1)
test_1D = np.random.rand(size,1)*0+1+ 1j * np.random.rand(size,1)*0 +1j

print((test_1D.shape)[0])

print(len(test_1D))

N = 200  # number of points in each direction of D-dimensional lattice  # if wavefunction is given: N = (wave.shape)[0]    # \\ [0] is arbitrary because quadratic matrix
A = 0.3  # spacing between lattice points   # assumption: A is input | can also be variant of L=N*a  
"""potential/hamiltonian"""
R = 36  # length from zero-point to potential-valleys 
M = 1   # mass of point particle
W = 1   # frequency
H_BAR = 1#!!! actually: 6.62607015*10**(-34)    # J*s
mu = (M*W*R**2)/H_BAR
epsilon = A/R

import scipy as sp 
from scipy import stats 
import matplotlib.pyplot as plt  


def gaussian_1D(sig): 
    phi = np.ones(N)
    for n, _ in np.ndenumerate(phi):
        x = (np.array(n) - mu) * A
        phi[n] = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.dot(x, x) / sig / 2) + 0j
    return phi
    
#print(gaussian_1D(1))

#x_data = np.arange(-100, 100, 200/N) 
  
## y-axis as the gaussian 
#y_data = stats.norm.pdf(x_data, 1, 0.5) 
  
#print(y_data)
#print(len(y_data))


## plot data 
#plt.plot(x_data, y_data)
#plt.show()
M=10000
FRAMES = 200


print(M/FRAMES)