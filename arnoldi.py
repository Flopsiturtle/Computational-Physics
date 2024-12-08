import numpy as np









N = 1000
b = np.diag(1/np.sqrt(np.arange(1,N)), k=1)
matrix1 = np.einsum('ji,jk', np.conjugate(b), b)


def norm(vector):
    return np.sqrt(np.vdot(vector,vector))

def matrix_multi(m, vector, iterations):
    for i in range(iterations):
        w = np.einsum('ij, j', m,vector)
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
        
def krylov_space(m, vector, eigenvalues):
    space = []
    for i in range(eigenvalues):
        space.append(matrix_multi(m, vector, i))
    return space 

def matrix_once(m, array):
    space = []
    for i in range(len(array)):
        space.append(matrix_multi(m, array[i], 1))
    return space 

def eigenvalues(array, m):
    space = []
    for i in range(len(array)):
        w = array[i]
        eigen = np.vdot(w, matrix_multi(m, w, 1))
        space.append(eigen)
    return space

def arnoldi(m, number_eigen, max_iter):
    v = np.ones(N)
    vectors = krylov_space(m, v, number_eigen)
    for count, i in enumerate(range(max_iter)):
        errors = []
        vectors = matrix_once(m, vectors)
        orth_vectors = gram_schmidt(vectors)
        eigen = eigenvalues(orth_vectors, m)
        for i in range(len(vectors)):
            LHS = matrix_multi(m, orth_vectors[i], 1)
            RHS = eigen[i]*orth_vectors[i]
            error = norm(LHS - RHS)
            errors.append(error)
        if np.all(np.array(errors) <  0.00001):
            print('nice')
            break
        else:
            #print(eigen)
            vectors = orth_vectors
            print(np.max(errors))
            continue
    
arnoldi(matrix1, 5, 100)    	




















    