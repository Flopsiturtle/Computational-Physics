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