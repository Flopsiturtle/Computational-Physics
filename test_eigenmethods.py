import numpy as np
import pandas as pd

import eigenmethods
import hamiltonian


def test_Hinv_inverse(shape_input,iterations,error_Hinv,maxiters_Hinv,mu,epsilon):      #### test works in multiple dimensions (Hinv and hamiltonian work in different dimensions)
    shape = shape_input.shape
    err = []
    fail = 0
    for i in range(iterations):
        v = np.random.rand(*shape)
        Hinv = eigenmethods.Hinv(v,error_Hinv,maxiters_Hinv,mu,epsilon)
        if isinstance(Hinv,str) is True:    # checking if Hinv is error because of maxiters
            fail = fail + 1
            err.append(0)
        else:   
            error = np.abs(v-hamiltonian.hamilton_variable(Hinv,mu,epsilon))   # calculating v - H*H^(-1)*v ~ 0
            err.append(np.max(error))
    return np.max(err),fail


def test_eigenvalue_vector(result_arnoldi):     # <-- do arnoldi one time before and use as input, then also below
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


#checks for some small deviation if we really have smallest eigenvalue
def test_ritz_method(result_arnoldi, number_eigen, start_deviation, iterations):   #starts with the given deviation and then makes it smaller each iteration
    eigen_values, eigen_vectors = result_arnoldi
    count = 0
    for i in range(iterations) :
        deviation = start_deviation / (i+1)
        v1 = np.ones_like(eigen_vectors[number_eigen])
        for j in range(len(eigen_vectors[number_eigen])):
            v1[j] = eigen_vectors[number_eigen][j] + deviation
        E1 = np.vdot(v1, hamiltonian.hamilton_variable(v1, mu, epsilon))
        if E1 <= eigen_values[number_eigen]:
            count += 1
    return count    #returns number of times the eigenenergie was smaller than the one we calculated



### test our results with linalgs programs (?)




##### Mickey/Flo H ## test with matrices we know eigenvalues/vectors -->> hamiltonians from different systems
#XXXXXXXX we decided not to because difficulty implementing into our system




## maybe animate evolution of different eigenstates? - Flo H   -- what do you mean? isnt that in convergence? - Flo T







''' --- run the code --- '''
############ he thought all these test ideas are good
######
############## for our tests in latex protocol, explain what variables chosen and also do tests with different parameters: shape, error_Hinv
######

#mu = 153.9     # our mu
mu = 20         # his mu
epsilon = 1/60





#v = np.ones(100)
#result_arnoldi = eigenmethods.arnoldi(v, 2,10**(-9),200,10**(-9),200,mu,epsilon)
#print(result_arnoldi)
#print((test_eigenvalue_vector(result_arnoldi)))
#exit()

#print(test_ritz_method(result_arnoldi,1,0.1,10))






exit()



''' testing Hinv '''

iterations = 20
grids = np.array([5, 10, 15])
print('Testing inverse of H*v ' + str(iterations) + ' times, for multiple N and D (tolerance =',10**(-7),', maxiters = 100). Maximum error: ')
tab = pd.DataFrame({'N': [], '1D': [], '2D': [], '3D': []})
for i in range(len(grids)):
    N = grids[i]
    dimensions = np.array([N,(N,N),(N,N,N)], dtype=object)
    lst = [N]
    for j in dimensions:
        psi = np.zeros(j)
        lst.append(str((test_Hinv_inverse(psi,iterations,10**(-7),100,mu,epsilon))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))

print(' ')

iterations = 100
grids = np.array([10**(-4), 10**(-7), 10**(-9)])
N = 15
psi = np.zeros(N)
print('Testing inverse of H*v ' + str(iterations) + ' times, for multiple tolerances and maxiters (N = 15, D = 1). Maximum error: ')
tab = pd.DataFrame({'tolerance': [], 'maxiter=10': [], 'maxiter=14': [], 'maxiter=20': [], 'maxiter=30': []})
for i in range(len(grids)):
    tolerance = grids[i]
    maxiters = np.array([10,14,20,30], dtype=object)
    lst = [tolerance]
    for j in maxiters:
        lst.append(str(((test_Hinv_inverse(psi,iterations,tolerance,j,mu,epsilon)))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))

print(' ')


''' testing eigenvalues/vectors '''

grids = np.array([20, 50, 100, 200])
print('Testing error of first four eigenvalues/vectors for multiple N and tolerances using starting vector np.ones (maxiters = 200). Maximum error: ')
tab = pd.DataFrame({'N': [], '10**(-3)': [], '10**(-5)': [],'10**(-7)': [],'10**(-9)': []})
for i in range(len(grids)):
    N = grids[i]
    lst = [N]
    v = np.ones(N)
    toler = np.array([10**(-3),10**(-5),10**(-7),10**(-9)])
    for tol in toler:
        arnoldi = eigenmethods.arnoldi(v, 4,tol,200,tol,200,mu,epsilon)
        lst.append(str((test_eigenvalue_vector(arnoldi))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))
#grids = np.array([20, 50, 100, 200])
#print('Testing error of first four eigenvalues/vectors for multiple N using starting vector np.ones and np.random (tolerances =',10**(-5),10**(-5),', maxiters = 300 300). Maximum error: ')
#tab = pd.DataFrame({'N': [], 'np.ones(N)': [], 'np.random.rand(N)': []})
#for i in range(len(grids)):
#    N = grids[i]
#    lst = [N]
#    typ = np.array([np.ones(N),np.random.rand(N)])
#    for v in typ:
#        arnoldi = eigenmethods.arnoldi(v, 4,10**(-5),300,10**(-5),300,mu,epsilon)
#        lst.append(str((test_eigenvalue_vector(arnoldi))))
#    tab.loc[len(tab)] = lst
#print(tab.to_string(index=False))

print(' ')

v = np.ones(100)
result_arnoldi = eigenmethods.arnoldi(v, 4,10**(-9),100,10**(-9),100,mu,epsilon)
result_arnoldi2 = eigenmethods.arnoldi(v, 10,10**(-9),100,10**(-9),100,mu,epsilon)
print('Choosing starting vector np.ones(100), tolerances',10**(-9),'maxiters = 100')
print('4 lowest eigenvalues',result_arnoldi[0], 'and the error for eigenvalue equation',test_eigenvalue_vector(result_arnoldi))
print('10 lowest eigenvalues',result_arnoldi2[0], 'and the error for eigenvalue equation',test_eigenvalue_vector(result_arnoldi2))

print(' ')


''' testing orthonormal '''

grids = np.array([20, 50, 100, 200])
print('Testing orthonormality of first 4 eigenvalues/vectors for multiple N and tolerances using starting vector np.ones (maxiters = 200). Maximum error: ')
tab = pd.DataFrame({'N': [], '10**(-3)': [], '10**(-5)': [],'10**(-7)': [],'10**(-9)': []})
for i in range(len(grids)):
    N = grids[i]
    lst = [N]
    v = np.ones(N)
    toler = np.array([10**(-3),10**(-5),10**(-7),10**(-9)])
    for tol in toler:
        arnoldi = eigenmethods.arnoldi(v, 4,tol,200,tol,200,mu,epsilon)
        lst.append(str(np.max(test_orthonormality(arnoldi[1]))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))

print(' ')


''' testing Ritz method'''

iterations = 100
grids = np.array(['first', 'second'])
print('Testing ritz method for first and second eigenvalue for multiple tolerances of arnoldi using starting vector np.ones(220) (start deviation Ritz: 0.01) (maxiters = 200). Number of times eigenvalue was NOT smallest out of 100 iterations of deviation: ')
tab = pd.DataFrame({'eigenvalue': [], '10**(-3)': [], '10**(-5)': [],'10**(-7)': [],'10**(-9)': []})
v = np.ones(220)
toler = np.array([10**(-3),10**(-5),10**(-7),10**(-9)])
arnoldi = []
for tol in toler:
    arnoldi.append(eigenmethods.arnoldi(v, 2,tol,200,tol,200,mu,epsilon))
for i in range(len(grids)):
    lst = [str(grids[i])]
    for j in range(len(toler)):
        lst.append(str((test_ritz_method(arnoldi[j],i,0.01,iterations))))
    tab.loc[len(tab)] = lst
print(tab.to_string(index=False))





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








