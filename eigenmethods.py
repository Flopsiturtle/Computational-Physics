import numpy as np

import variables
import hamiltonian


""" --- conjugate gradient --- """
# A x = b
# H x = v -> x = H^-1 v     # so conjugate gradient gives x as result which is our wanted quantity
# now: implement hamiltonian in function itself

def Hinv(v,tolerance,maxiters):
    size = (v.shape)[0]
    D = len(v.shape)    
    x0 = np.zeros((size,)*D)   # D-dimensional --- can be changed to x0 = np.zeros(N) if only works in 1D
    r0 = v - hamiltonian.hamilton(x0)
    if np.max(r0) <= tolerance:    
        return x0
    p0 = r0
    for i in np.arange(1,maxiters+1):
        alpha0 = (np.vdot(r0,r0)) / (np.vdot(p0,hamiltonian.hamilton(p0)))    
        x = x0 + alpha0*p0
        r = r0 - alpha0*hamiltonian.hamilton(p0)
        if np.max(r) <= tolerance: 
            return x    # return x,i if you want iterations in output
        beta = (np.vdot(r,r)) / (np.vdot(r0,r0))
        p0 = r + beta*p0
        x0 = x
        r0 = r



''' --- Arnoldi method --- '''

def norm(vector):
    return np.sqrt(np.vdot(vector,vector))

def matrix_multi(vector, iterations, error_Hinv, maxiters_Hinv):
    for i in range(iterations):
        w = Hinv(vector, error_Hinv, maxiters_Hinv)
        vector = w
    return vector

def gram_schmidt(array):
    space = []
    for i in range(len(array)):
        w = array[i]
        a = w
        for j in range(len(space)):
            a += -np.vdot(space[j], w)*space[j]
        a = a/norm(a)
        space.append(a)
    return space
        
def krylov_space(vector, number_eigen, error_Hinv, maxiters_Hinv):
    space = []
    for i in range(number_eigen):
        space.append(matrix_multi(vector, i, error_Hinv, maxiters_Hinv))
    return space 

def matrix_once(array, error_Hinv, maxiters_Hinv):
    space = []
    for i in range(len(array)):
        space.append(matrix_multi(array[i], 1, error_Hinv, maxiters_Hinv))
    return space 

def eigenvalues(array, error_Hinv, maxiters_Hinv):
    space = []
    for i in range(len(array)):
        w = array[i]
        eigen = np.vdot(w, matrix_multi(w, 1, error_Hinv, maxiters_Hinv))
        space.append(eigen)
    return space

def arnoldi(v, number_eigen, error_arnoldi, maxiter_arnoldi, error_Hinv, maxiters_Hinv):
    vectors = krylov_space(v, number_eigen, error_Hinv, maxiters_Hinv)
    for count, i in enumerate(range(maxiter_arnoldi)):
        errors = []
        vectors = matrix_once(vectors, error_Hinv, maxiters_Hinv)
        orth_vectors = gram_schmidt(vectors)
        eigen = eigenvalues(orth_vectors, error_Hinv, maxiters_Hinv)
        for i in range(len(vectors)):
            LHS = matrix_multi(orth_vectors[i], 1, error_Hinv, maxiters_Hinv)
            RHS = eigen[i]*orth_vectors[i]
            error = norm(LHS - RHS)
            errors.append(error)
        if np.all(np.array(errors)) <  error_arnoldi:
            return 1/np.array(eigen),orth_vectors
            #print('Found eigenvalues')
        else:           # same as mine: do we have to do extra step for <maxiters or output "None" enough?           
            vectors = orth_vectors
            #print(np.max(errors))




""" --- test the code --- """
v = np.ones(200)      # works only for high N!! # but also not N much larger than 200 !!
#print(arnoldi(v,5,0.0001,100,0.0001,100))   


# tests for arbitrary v
#### 1D test

v = np.ones(10)   

error = 0.00001
max_integers = 200
#print(Hinv(v,error,max_integers))


#### 1D test for complex with our gaussian
#n, v=variables.gaussian_1D(-int(N/4),int(N/16))
v = variables.normalize(v)
#print(Hinv(v,error,max_integers))


# 2D test    ######## fixed conjugate, works now in multidimensional!!!
v = np.ones((20,20))   
error = 0.00001
max_integers = 200
#print(Hinv(v,error,max_integers))
