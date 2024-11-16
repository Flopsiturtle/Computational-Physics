# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:33:12 2024

@author: Florian Hollants
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import scipy.constants as const



def initialize(gauss = False, random = False):
    """declaring variables and initializing Psi"""
    global H_BAR, MASS, D, N, L, A
    H_BAR = .001
    MASS = 1
    D = 1
    N = 3000
    L = 2
    A = L/N
    global OMEGA, R, T, M
    OMEGA=1
    R=L/4
    T=1
    M=1000
    global MU, EPSILON, TAU, sig, mu
    MU=MASS*OMEGA*R**2/H_BAR
    EPSILON = A/R
    TAU=OMEGA*T/M #
    sig = R/10
    mu = int(N/2)
    global shape, origin, Psi
    shape = (N,)  # tuple of D N's to shape ndarrays correctly
    origin = (0,)  # tuple of D zeros to call position in grid later
    for _ in range(D-1):
        shape += (N,)
        origin += (0,)
    
    Psi =np.arange(N**D).reshape(shape).astype('complex')*0+1
    
    if random:
        Psi = np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
        return()
    
    """
    if gauss:
        for n, _ in np.ndenumerate(Psi):
            x = (np.array(n) - mu) * A
            Psi[n] = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.dot(x, x) / sig / 2) + 0j
        return()"""
    return()


initialize(gauss = False, random = False)
#shifting the origin to the center of the grid
V=np.empty_like(Psi)
for n, _ in np.ndenumerate(Psi):
    V[n]=np.dot(np.array(n)-int(N/2),np.array(n)-int(N/2))



def potential():
    """defines the potential"""
    V=np.empty_like(Psi)*0
    for n, _ in np.ndenumerate(Psi):
        index_arr = np.array(n)
        V[n]=MU/8*(EPSILON**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2
    return V


def inner_product(func1, func2):
    """calculates the inner product of two arrays"""
    return np.sum(np.multiply(np.conjugate(func1),func2))


def normalize(func):
    """normalizes input function"""
    return func*1/np.sqrt(inner_product(func,func))


def laplace(func):
    """calculating the laplacian of ndarray"""
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap



def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*MU*EPSILON**2)*laplace(func)+potential()*func


def time_evol(func, n):
    """time evolution using second-order Fourier transform"""
    for _ in range(n):
        func = func - 1j*hamilton(func)*TAU-TAU**2*hamilton(hamilton(func))/2
    return func

def check_linearity(func):
    res = 0
    n = 100
    for _ in range(n):
        Psi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
        Phi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
        res += np.abs(np.max(np.abs(func(2*Psi+Phi)-2*func(Psi)-func(Phi))))
    if res < 1/10**9:
        print(func.__name__ + " is linear")
        return res/n
    else:
        print(func.__name__ + " is not linear")
        print(res)
        return res/n


def check_hermitian(func):
    res = 0
    n = 100
    for _ in range(n):
        Psi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
        res += np.abs(inner_product(func(Psi),Psi)-inner_product(Psi,func(Psi)))
    if res < 1/10**9:
        print(func.__name__ + " is hermitian")
        return res/n
    else:
        print(func.__name__ + " is not hermitian")
        return res/n


def check_positivity(func):
    count = 0
    n = 100
    for _ in range(n):
        Psi=np.random.uniform(-1,1,shape)+1j*np.random.uniform(-1,1,shape)
        if inner_product(Psi,func(Psi)).real < 0:
            count += 1
    print("The Energy has been negative " + str(count) + " out of " + str(n) + " times")
    return count


def check_eigen():
    k = [np.random.uniform(-1,1)+1j*np.random.uniform(-1,1) for _ in range(D)]
    for n, _ in np.ndenumerate(Psi):
        Psi[n]=np.exp(2j*np.pi*np.dot(np.array(n)-int(N/2),k)/N)
    E = D/(MU*EPSILON**2)
    for _ in range(D):
        E += -1/(2*MU*EPSILON**2)*(Psi[int(N/2)+1]+Psi[int(N/2)-1])
    print(E)
    return()
#print(np.eye(1, D, 0))

#check_eigen()


Psi = normalize(Psi)



fig, ax = plt.subplots()

line2 = ax.plot(np.linspace(-L/(2*R), L/(2*R), N), np.abs(Psi)**2/np.sqrt(EPSILON))[0]
ax.set(xlim=[-L/(2*R), L/(2*R)], ylim=[-1, 1])



t = [0]
Norm = [inner_product(Psi, Psi).real]
E = [inner_product(Psi, hamilton(Psi)).real]



running = Psi

def update(frame):
    # for each frame, update the data stored on each artist.
    # update the line plot:
    global running
    global t
    global Norm
    global E
    running = time_evol(running, 10)
    line2.set_ydata(np.abs(running)**2/np.sqrt(EPSILON))
    t.append(len(t)*10)
    Norm.append(inner_product(running, running))
    E.append(inner_product(running, hamilton(running)))
    return (line2)


ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=30)
plt.show()

plt.plot(t, Norm, E)
#check_linearity(hamilton)
#check_hermitian(hamilton)
#check_positivity(hamilton)


#print(Psi)
#print(inner_product(hamilton(Psi),Psi)-inner_product(Psi,hamilton(Psi)))
#print(max(np.abs(hamilton(2*Psi+Phi)-2*hamilton(Psi)-hamilton(Phi))))
#print(laplace(Psi))
#print(hamilton(Psi))
#print(time_evol(Psi, 20))
#print(inner_product(time_evol(Psi, 20), time_evol(Psi, 20)))


"""
def dependence(func):
    global M
    taus = []
    E_so = []
    E_st = []
    norms = []
    diff = []
    for m in range(100):
        M = 10*m
        taus.append(tau)
        E_so.append(inner_product(so_integrator(func, 10), hamilton(so_integrator(func, 10)))/inner_product(so_integrator(func, 10), so_integrator(func, 10)))
        E_st.append(inner_product(Strang_Splitting(func, 10), hamilton(Strang_Splitting(func, 10)))/inner_product(Strang_Splitting(func, 10), Strang_Splitting(func, 10)))
        norms.append(inner_product(so_integrator(func, 10), so_integrator(func, 10)))
        diff.append(np.average(np.abs(so_integrator(func,10)-Strang_Splitting(func,10))))
    print(taus)
    return taus, E_so, E_st, norms, diff"""