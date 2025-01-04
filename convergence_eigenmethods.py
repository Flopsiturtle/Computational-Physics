import numpy as np
import matplotlib.pyplot as plt
import variables
import eigenmethods


#print(eigenmethods.arnoldi(np.ones(190), 4, 0.000001, 500, 0.000001, 500)[0])


#### !!!!!!!for some N we get an error!!!!!!!


#def get_eigen_array(number_eigen,start,finish,gap):
#    store_eigen = []
#    for N in np.arange(start,finish+gap,gap):
#        v = np.ones(N)
#        #print(eigenmethods.arnoldi(v, 1, 0.0001, 100, 0.0001, 200)[0])
#        data = eigenmethods.arnoldi(v, number_eigen, 0.0001, 500, 0.0001, 500)[0]
#        if isinstance(data,str) is False: 
#            store_eigen.append((data,N))
#    return store_eigen
#a = get_eigen_array(1,50,100,10)
#x=[]
#y=[]
#for i in np.arange(len(a)):
#    x.append(a[i][1])
#    y.append(a[i][0])



def get_data(number_eigen,start,finish,gap,mu,epsilon):
    x = []
    y = []
    for N in np.arange(start,finish+gap,gap):
        v = np.random.random(N)
        #print(eigenmethods.arnoldi(v, 1, 0.0001, 100, 0.0001, 200)[0])
        data = eigenmethods.arnoldi(v, number_eigen, 10**(-7), 500, 10**(-7), 500,mu,epsilon)[0]
        if isinstance(data,str) is False: 
            x.append(N)
            y.append(data)
    return x,y


#mu = 153.9     # our mu
mu = 20         # his mu
epsilon = 1/60
"""
data = get_data(4,40,240,40,mu,epsilon)     ### maybe delete input with start,finish,gap to be always the same and give input of errors and maxiters, depending on usecase we want for final test
plt.plot(data[0],data[1],'ro')
plt.show()
"""
### i think problems with eigenvalues might be fixed now because of change in potential???


## for N = 200 we plot the eigenvalues against the tolerance of the arnoldi method (not the H_inv)
## we also blor the evolution of the eigenvectors with the tolerance
## my guess is that we will get eigensetates for the left and right valley seperately and thats whats fucking us over
def tol_evol(number_eigen):
    v = np.random.random(200)
    values = []
    vectors = []
    x = []
    for tol in range(1,7):
        A, B = eigenmethods.arnoldi(v, number_eigen, 10**(-tol), 200, 10**(-6), 200)
        print(A)
        values.append(A)
        vectors.append(B)
        x.append(tol)
    figure, axs = plt.subplots(number_eigen,7)
    for j in range(number_eigen):
        axs[j,0].plot(x,np.array(values).transpose()[j])
        for i in range(1,7):
            axs[j,i].plot(np.arange(-100,100)*variables.epsilon,np.array(vectors)[i-1,j])



def plot_eigenvectors(eigenvalues, eigenvectors):
    num_eigenvectors = len(eigenvalues)
    cols = 2
    rows = (num_eigenvectors + cols - 1) // cols

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
        ax = axes[i]
        ax.plot(np.arange(-100,101)*variables.epsilon, eigenvector, marker='o', linestyle='-')
        ax.set_title(f"Eigenvalue: {eigenvalue:.4f}", fontsize=14)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def generate_parity(N, mu, epsilon):
    v = np.concatenate((np.random.random(N//2),[0])) 
    v_even = np.concatenate((v,list(reversed(v[:-1]))))
    v_odd = np.concatenate((-v,list(reversed(v[:-1]))))
    Eig_val_even, Eig_vec_even = eigenmethods.arnoldi(v_even, 2, 10**(-6), 500, 10**(-6), 500,mu,epsilon)
    Eig_val_odd, Eig_vec_odd = eigenmethods.arnoldi(v_odd, 2, 10**(-6), 500, 10**(-6), 500,mu,epsilon)
    return np.concatenate((Eig_val_even,Eig_val_odd)), np.concatenate((Eig_vec_even,Eig_vec_odd))
"""
v = np.random.random(201)
Eig_val, Eig_vec = eigenmethods.arnoldi(v, 4, 10**(-6), 500, 10**(-6), 500,mu,epsilon)
plot_eigenvectors(Eig_val, Eig_vec)
plot_parity()
"""

def continuum():
    multi = np.arange(1,11, 2)
    Eig_val, Eig_vec = [], []
    for m in multi:
        N = 201 * m
        eps = epsilon/m
        v = np.concatenate((np.random.random(N//2),[0])) 
        v_even = np.concatenate((v,list(reversed(v[:-1]))))
        v_odd = np.concatenate((-v,list(reversed(v[:-1]))))
        print(m, N, eps)
        print(v_even)
        Eig_val_even, Eig_vec_even = eigenmethods.arnoldi(v_even, 2, 10**(-6), 500, 10**(-6), 500,mu,eps)
        Eig_val_odd, Eig_vec_odd = eigenmethods.arnoldi(v_odd, 2, 10**(-6), 500, 10**(-6), 500,mu,eps)
        Eig_val.append(np.concatenate((Eig_val_even,Eig_val_odd)))
        Eig_vec.append(np.concatenate((Eig_vec_even,Eig_vec_odd)))
    return Eig_val, Eig_vec

plt.plot(np.arange(1,11, 2), continuum()[0])
plt.show()
#tol_evol(4)
### choose a value for N for which we have already converged in the infinite volume limit
### lets say N_0 = 200 for epsilon_0 = 1/60
### set up a series for the loop maybe N=5^kN_0 and epsilon=epsilon_0/5^k for k from a to b
### then we run eigenmethods.arnoldi for these values of the variables


