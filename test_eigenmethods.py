import numpy as np

import eigenmethods
import hamiltonian


######
############## for our tests in latex protocol, explain what variables chosen and also do tests with different parameters: shape, error_Hinv
######


''' --- #Flo T ### test if inverse is correct --- '''
#### test works in multiple dimensions (Hinv and hamiltonian in different dimensions)
def test_Hinv_inverse(shape_input,iterations,error_Hinv,maxiters_Hinv):
    shape = shape_input.shape
    err = []
    for i in range(iterations):
        v = np.random.rand(*shape)
        error = np.abs(v-hamiltonian.hamilton(eigenmethods.Hinv(v,error_Hinv,maxiters_Hinv)))   # calculating v - H*H^(-1)*v ~ 0
        err.append(error)
    return np.max(err)

#test 1D
print(test_Hinv_inverse(np.ones(50),10,0.0001,100))
#test 2D
print(test_Hinv_inverse(np.ones((50,50)),10,0.0001,100))



''' --- #Flo T ## test if really eigenvalues/vectors --- '''
def test_eigenvalue_vector(result_arnoldi):   
    # habe jetzt entschieden das als input zu nehmen, da wir letzlich ja andere Tests vorher machen können um zu testen ob immer die Eigenwerte rauskommen
    # sonst müssen wir jedes mal die arnoldi Funktion bei den Tests ausführen, so führen wir sie einmal vor allen Tests aus
    size = len(result_arnoldi[0])
    err = []
    for i in np.arange(size):
        LHS = hamiltonian.hamilton(result_arnoldi[1][i])
        RHS = result_arnoldi[0][i]*result_arnoldi[1][i]
        max_error = np.max(np.abs(LHS-RHS))
        err.append(max_error)
    return err


v = np.ones(200)
#result_arnoldi = eigenmethods.arnoldi(v,5,0.0001,100,0.0001,100)
print(test_eigenvalue_vector(eigenmethods.arnoldi(v,5,10**(-7),200,10**(-7),200)))

################## the errors increase for higher eigenvalues????? -> can we change our breaking point in arnoldi method so that highest calculated eigenvalue has our wanted error?
########## he says he wants an error <= 1% for eigenvalues/vectors


def test_orthonormality(vectors):
    #sollte klappen mit den richtigen inputs, ich konnte es jetzt aber noch nicht für arnoldi testen
    #output ist error matrix, idk wie oder ob das in mehr als einer dimension klappt
    matrix = np.column_stack(vectors)
    gram = np.dot(matrix.T, matrix)
    identity = np.eye(len(vectors))
    return np.abs(gram-identity)

#test für zwei orthonormale vektoren
#vec = [np.array([1,1])/np.sqrt(2), np.array([1,-1])/np.sqrt(2)]
print(np.max(test_orthonormality(eigenmethods.arnoldi(v,5,0.0001,100,0.0001,100)[1])))






###### Tests:
#Flo H ## test if really orthonormal
#Mickey ## test if really lowest eigenvalue
#xxx ## test our results with linalgs programs (?)

#Mickey/Flo H ## test with matrices we know eigenvalues/vectors -->> hamiltonians from different systems
### .....

############ he thought all these test ideas are good

### maybe animate evolution of different eigenstates? - Flo H

### some kind of test for checking if chosen error and maxiters is good?    # e.g. test different errors and outcomes   

### also if program is run multiple times, is convergence the same? what error if not --> well depends on chosen error of course...



