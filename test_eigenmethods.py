import numpy as np
import pandas as pd

import eigenmethods
import hamiltonian



''' --- defining different tests --- '''
def test_Hinv_inverse(shape_input,iterations,error_Hinv,maxiters_Hinv,mu,epsilon):      #### test works in multiple dimensions (Hinv and hamiltonian work in different dimensions)
    shape = shape_input.shape
    err = []
    fail = 0
    for i in range(iterations):
        v = np.random.rand(*shape)
        Hinv = eigenmethods.Hinv(v,error_Hinv,maxiters_Hinv,mu,epsilon)
        if isinstance(Hinv,str) is True:    # checking if Hinv outputs an error because of reached maxiters; add to count if true
            fail = fail + 1
            err.append(0)
        else:   
            error = np.abs(v-hamiltonian.hamilton_variable(Hinv,mu,epsilon))   # calculating v - H*H^(-1)*v ~ 0
            err.append(np.max(error))
    return np.max(err),fail


def test_eigenvalue_vector(result_arnoldi):     # <-- do arnoldi one time before and use as input, then also below in tests
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



def test_ritz_method(result_arnoldi, number_eigen, start_deviation, iterations):   #checks for some small deviation if we really have smallest eigenvalue
    eigen_values, eigen_vectors = result_arnoldi
    count = 0
    for i in range(iterations) :
        deviation = start_deviation / (i+1)         #starts with the given deviation and then makes it smaller each iteration
        v1 = np.ones_like(eigen_vectors[number_eigen])
        for j in range(len(eigen_vectors[number_eigen])):
            v1[j] = eigen_vectors[number_eigen][j] + deviation
        E1 = np.vdot(v1, hamiltonian.hamilton_variable(v1, mu, epsilon))
        if E1 <= eigen_values[number_eigen]:
            count += 1
    return count    #returns number of times the eigenenergy was smaller than the one we calculated



''' --- define different parameter-tests which output tabulars and can be called separately --- '''

''' testing Hinv '''
def tab_Hinv_N(N_array):
    iterations = 20
    grids = N_array
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

def tab_Hinv_tol(tolerance_array):
    iterations = 100
    grids = tolerance_array
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


''' testing eigenvalues/vectors '''
def tab_eigen_ones(N_array):
    grids = N_array
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
"""
####### commented out because we didnt use in final report but still used for testing and finding best v
def tab_eigen_random(N_array):      
    grids = N_array
    print('Testing error of first four eigenvalues/vectors for multiple N using starting vector np.ones and np.random (tolerances =',10**(-5),10**(-5),', maxiters = 300 300). Maximum error: ')
    tab = pd.DataFrame({'N': [], 'np.ones(N)': [], 'np.random.rand(N)': []})
    for i in range(len(grids)):
        N = grids[i]
        lst = [N]
        typ = np.array([np.ones(N),np.random.rand(N)])
        for v in typ:
            arnoldi = eigenmethods.arnoldi(v, 4,10**(-5),300,10**(-5),300,mu,epsilon)
            lst.append(str((test_eigenvalue_vector(arnoldi))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))
"""
def tab_eigen_even(N_array):
    grids = N_array
    print('Testing error of first two even eigenvalues/vectors for multiple N and tolerances using ones as starting vector (maxiters = 10000). Maximum error: ')
    tab = pd.DataFrame({'N': [], '10**(-3)': [], '10**(-5)': [],'10**(-7)': []})
    for i in range(len(grids)):
        N = grids[i]
        lst = [N]
        v = np.concatenate((np.ones(N//2),[0]))
        v_even = np.concatenate((v,list(reversed(v[:-1]))))
        toler = np.array([10**(-3),10**(-5),10**(-7)])
        for tol in toler:
            arnoldi = eigenmethods.arnoldi(v_even, 2,tol,10000,tol,10000,mu,epsilon)
            #lst.append(arnoldi[0])
            lst.append(str((test_eigenvalue_vector(arnoldi))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))

def tab_eigen_odd(N_array):
    grids = N_array
    print('Testing error of first two odd eigenvalues/vectors for multiple N and tolerances using ones as starting vector (maxiters = 10000). Maximum error: ')
    tab = pd.DataFrame({'N': [], '10**(-3)': [], '10**(-5)': [],'10**(-7)': []})
    for i in range(len(grids)):
        N = grids[i]
        lst = [N]
        v = np.concatenate((np.ones(N//2),[0]))
        v_odd = np.concatenate((-v,list(reversed(v[:-1]))))
        toler = np.array([10**(-3),10**(-5),10**(-7)])
        for tol in toler:
            arnoldi = eigenmethods.arnoldi(v_odd, 2,tol,10000,tol,10000,mu,epsilon)
            lst.append(str((test_eigenvalue_vector(arnoldi))))
        tab.loc[len(tab)] = lst
    print(tab.to_string(index=False))

def eigen_number():
    v = np.ones(100)
    result_arnoldi = eigenmethods.arnoldi(v, 4,10**(-9),100,10**(-9),100,mu,epsilon)
    result_arnoldi2 = eigenmethods.arnoldi(v, 10,10**(-9),100,10**(-9),100,mu,epsilon)
    print('Choosing starting vector np.ones(100), tolerances',10**(-9),'maxiters = 100')
    print('4 lowest eigenvalues',result_arnoldi[0], 'and the error for eigenvalue equation',test_eigenvalue_vector(result_arnoldi))
    print('10 lowest eigenvalues',result_arnoldi2[0], 'and the error for eigenvalue equation',test_eigenvalue_vector(result_arnoldi2))


''' testing orthonormal '''
def tab_ortho(N_array):
    grids = N_array
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


''' testing Ritz method'''
def tab_ritz(iterations):
    grids = np.array(['first', 'second'])
    print('Testing ritz method for first and second eigenvalue for multiple tolerances of arnoldi using starting vector np.ones(220) (start deviation Ritz: 0.01) (maxiters = 200). Number of times eigenvalue was NOT smallest out of '+str(iterations)+' iterations of deviation: ')
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



''' --- run the code --- '''
#mu = 153.9     # our mu from project 1.1
mu = 20         # given mu
epsilon = 1/60  # our epsilon from project 1.1

tab_Hinv_N(N_array = np.array([5, 10, 15]))
print(' ')
tab_Hinv_tol(tolerance_array = np.array([10**(-4), 10**(-7), 10**(-9)]))
print(' ')
tab_eigen_ones(N_array = np.array([20, 50, 100, 200]))
print(' ')
tab_eigen_even(N_array = np.array([40, 80, 120, 200]))
print(' ')
tab_eigen_odd(N_array = np.array([40, 80, 120, 200]))
print(' ')
eigen_number()
print(' ')
tab_ortho(N_array = np.array([20, 50, 100, 200]))
print(' ')
tab_ritz(iterations = 100)
