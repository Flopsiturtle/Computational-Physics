import numpy as np
import matplotlib.pyplot as plt
import variables
import eigenmethods


def generate_parity(N, mu, epsilon):
    v = np.concatenate((np.ones(N//2),[0])) 
    v_even = np.concatenate((v,list(reversed(v[:-1]))))
    v_odd = np.concatenate((-v,list(reversed(v[:-1]))))
    Eig_val_even, Eig_vec_even = eigenmethods.arnoldi(v_even, 2, 10**(-7), 10000, 10**(-7), 10000,mu,epsilon)
    Eig_val_odd, Eig_vec_odd = eigenmethods.arnoldi(v_odd, 2, 10**(-7), 10000, 10**(-7), 10000,mu,epsilon)
    return np.concatenate((Eig_val_even,Eig_val_odd)), np.concatenate((Eig_vec_even,Eig_vec_odd))


def inf_vol(start, finish, gap, mu, epsilon):
    Eig_val, Eig_vec = [], []
    print('N')
    for N in np.arange(start,finish+gap,gap):
        print(N)
        val, vec = generate_parity(N, mu, epsilon)
        Eig_val.append(val)
        Eig_vec.append(vec)
    return Eig_val, Eig_vec



def continuum():
    multi = np.arange(1,11, 2)
    Eig_val, Eig_vec = [], []
    print('m , N , epsilon')
    for m in multi:
        N = 241 * m
        eps = epsilon/m
        print(m, N, eps)
        val, vec = generate_parity(N, mu, eps)
        Eig_val.append(val)
        Eig_vec.append(vec)
    return Eig_val, Eig_vec

def plot_inf_vol():
    #RUN THIS TO GET THE INFINITE VOLUME LIMIT
    plt.plot(np.arange(40,280+40,40), inf_vol(40, 280, 40, mu, epsilon)[0], 'ro')
    plt.title('Infinite Volume Limit', fontsize = 40, weight='bold')
    plt.xlabel('N', fontsize = 40)
    plt.ylabel(r'$\frac{E}{\hbar\omega}$', fontsize = 40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return()

def plot_cont_lim():
    #RUN THIS TO GET THE CONTINUUM LIMIT (might take a little bit)
    plt.plot(np.arange(1,11, 2), continuum()[0], 'ro')
    plt.title('Continuum Limit', fontsize = 40, weight = 'bold')
    plt.xlabel('multiplier m', fontsize = 40)
    plt.ylabel(r'$\frac{E}{\hbar\omega}$', fontsize = 40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return()

def plot_eigenvectors(N):
    eigenvalues, eigenvectors = generate_parity(N, mu, epsilon)
    num_eigenvectors = len(eigenvalues)
    cols = 2
    rows = (num_eigenvectors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()
    
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
        ax = axes[i]
        ax.plot(np.arange(-len(eigenvector)//2,len(eigenvector)//2)*variables.epsilon, eigenvector, marker='o', linestyle='-')
        ax.set_title(f"Eigenvalue: {eigenvalue:.4f}", fontsize=40, weight = 'bold')
        ax.set_xlabel(r'$\frac{x}{r}$', fontsize = 40)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    print(eigenvalues)

#------ run the code -----------

#mu = 153.9     # our mu
mu = 20         # his mu
epsilon = 1/60



#plot_inf_vol()
#plot_cont_lim()
#plot_eigenvectors(N=41)

