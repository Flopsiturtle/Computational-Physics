import numpy as np

import eigenmethods
import hamiltonian


def test_Hinv_inverse(shape_input,iterations,error_Hinv,maxiters_Hinv,mu,epsilon):      #### test works in multiple dimensions (Hinv and hamiltonian in different dimensions)
    shape = shape_input.shape
    err = []
    for i in range(iterations):
        v = np.random.rand(*shape)
        error = np.abs(v-hamiltonian.hamilton_variable(eigenmethods.Hinv(v,error_Hinv,maxiters_Hinv,mu,epsilon),mu,epsilon))   # calculating v - H*H^(-1)*v ~ 0
        err.append(error)
    return np.max(err)


def test_eigenvalue_vector(result_arnoldi):     # <-- do arnoldi one time before and use as input
    size = len(result_arnoldi[0])
    err = []
    for i in np.arange(size):
        LHS = hamiltonian.hamilton_variable(result_arnoldi[1][i],mu,epsilon)
        RHS = result_arnoldi[0][i]*result_arnoldi[1][i]
        max_error = np.max(np.abs(LHS-RHS))
        err.append(max_error)
    return err


def test_orthonormality(vectors):
    matrix = np.column_stack(vectors)
    gram = np.dot(matrix.T, matrix)
    identity = np.eye(len(vectors))
    return np.abs(gram-identity)



#Mickey ## test if really lowest eigenvalue




## test our results with linalgs programs (?)




##### Mickey/Flo H ## test with matrices we know eigenvalues/vectors -->> hamiltonians from different systems
##### xxx we decided not to




## maybe animate evolution of different eigenstates? - Flo H   -- what do you mean? isnt that in convergence? - Flo T




# some kind of test for checking if chosen error and maxiters is good?    # e.g. test different errors and outcomes   





###### also if program is run multiple times, is convergence the same? what error if not --> well depends on chosen error of course...




''' --- test the code --- '''
############ he thought all these test ideas are good
######
############## for our tests in latex protocol, explain what variables chosen and also do tests with different parameters: shape, error_Hinv
######


#mu = 153.9     # our mu
mu = 20         # his mu
epsilon = 1/60







exit()


#test inverse 1D
print(test_Hinv_inverse(np.ones(200),20,10**(-7),100,mu,epsilon))
#test inverse 2D
#print(test_Hinv_inverse(np.ones((50,50)),10,10**(-9),100))



v = np.ones(200)
result_arnoldi = eigenmethods.arnoldi(v,5,10**(-9),200,10**(-9),200,mu,epsilon)
print(result_arnoldi[0])

#test eigenvalues/vectors
print(test_eigenvalue_vector(result_arnoldi))
################## the errors increase for higher eigenvalues????? -> can we change our breaking point in arnoldi method so that highest calculated eigenvalue has our wanted error?
########## he says he wants an error <= 1% for eigenvalues/vectors
###### maybe not here explicitly eigenvalue error 1

#test für zwei orthonormale vektoren
#vec = [np.array([1,1])/np.sqrt(2), np.array([1,-1])/np.sqrt(2)]
print(np.max(test_orthonormality(result_arnoldi[1])))








