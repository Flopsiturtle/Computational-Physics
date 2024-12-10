import numpy as np

import eigenmethods
import hamiltonian


''' --- #Flo T ### test if inverse is correct --- '''
def test_Hinv_inverse(v,error_Hinv,maxiters_Hinv):
    err = np.abs(hamiltonian.hamilton(v)*eigenmethods.Hinv(v,error_Hinv,maxiters_Hinv))   # ist das wirklich richtig für inverse??
    return err

v=np.ones(200)
#print(test_Hinv_inverse(v,0.0001,100))

v=np.ones((50,50))
#print(test_Hinv_inverse(v,0.0001,100))



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
print(test_eigenvalue_vector(eigenmethods.arnoldi(v,5,0.0001,100,0.0001,100)))







###### Tests:
#Flo H ## test if really orthonormal
#Mickey ## test if really lowest eigenvalue
#xxx ## test our results with linalgs programs (?)

#Mickey/Flo H ## test with matrices we know eigenvalues/vectors -->> hamiltonians from different systems
### .....


### some kind of test for checking if chosen error and maxiters is good?    # e.g. test different errors and outcomes   

### also if program is run multiple times, is convergence the same? what error if not --> well depends on chosen error of course...



