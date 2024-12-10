import numpy as np
import Eigenmethods
from variables import *

#!!!!!!!
######### i took arnoldi from here to eigenmethods and changed it for modularity!!!!!!!!
#!!!!!!!

def norm(vector):
    return np.sqrt(np.vdot(vector,vector))

def matrix_multi(vector, iterations):
    for i in range(iterations):
        w = Eigenmethods.Hinv(vector, 0.01, 200)[0]
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
        
def krylov_space(vector, eigenvalues):
    space = []
    for i in range(eigenvalues):
        space.append(matrix_multi(vector, i))
    return space 

def matrix_once(array):
    space = []
    for i in range(len(array)):
        space.append(matrix_multi(array[i], 1))
    return space 

def eigenvalues(array):
    space = []
    for i in range(len(array)):
        w = array[i]
        eigen = np.vdot(w, matrix_multi(w, 1))
        space.append(eigen)
    return space

def arnoldi(number_eigen, max_iter):
    v = np.ones(N)
    vectors = krylov_space(v, number_eigen)
    for count, i in enumerate(range(max_iter)):
        errors = []
        vectors = matrix_once(vectors)
        orth_vectors = gram_schmidt(vectors)
        eigen = eigenvalues(orth_vectors)
        for i in range(len(vectors)):
            LHS = matrix_multi(orth_vectors[i], 1)
            RHS = eigen[i]*orth_vectors[i]
            error = norm(LHS - RHS)
            errors.append(error)
        if np.all(np.array(errors) <  0.00001):
            print(1/np.array(eigen))
            print('nice')
            break
        else:
            vectors = orth_vectors
            #print(np.max(errors))
            continue
arnoldi(5, 100)   
arnoldi(10, 100)   
arnoldi(15, 100)   





















    