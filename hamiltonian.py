import numpy as np

def hamiltonian_lattice(phi,mu,epsilon):
    D = phi.shape
    H = np.zeros(D)



test=np.array([[[1,2,6],[2,3,7]],[[2,4,6],[3,4,9]]])

print(test)
print(np.roll(test,1,0))

# axis 2 -> x
# axis 1 -> y
# axis 0 -> z








#print(np.zeros((2,2,2,2)))