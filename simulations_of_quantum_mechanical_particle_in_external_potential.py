# Authors: Florian Telleis, Florian Hollants, Mickey Wilke

import numpy as np
from scipy.fftpack import fft, ifft
from scipy import stats 
import matplotlib.pyplot as plt
import matplotlib.animation as animation




L = 60
N = 200 # number of points in each direction of D-dimensional lattice  # if wavefunction is given: N = (wave.shape)[0]    # \\ [0] is arbitrary because quadratic matrix
A = L/N  # spacing between lattice points   # assumption: A is input | can also be variant of L=N*a  

R = 18  # length from zero-point to potential-valleys 
Mass = 0.475   # mass of point particle
W = 1   # frequency
H_BAR = 1#!!! actually: 6.62607015*10**(-34)    # J*s

mu = (Mass*W*R**2)/H_BAR
epsilon = A/R


T = 10     # time
M = 1000*T  # large value
tau = W*T/M  # time step

FRAMES = 200    # number of frames in final animation
FPS = int(FRAMES/T)     # number of frames per second if given time T is real time in seconds
#FPS = 23




def gaussian_1D(mean,sigma): 
    x_data = np.arange(-int(N/2), int(N/2)) 
    y_data = stats.norm.pdf(x_data, mean, sigma)*np.exp(-5j*x_data) 
    return x_data, y_data 


def inner_product(func1, func2):
    """calculates the inner product of two arrays"""
    return np.dot(np.conjugate(func1),func2)


def normalize(func):
    """normalizes input function"""
    return func*1/np.sqrt(inner_product(func,func))


def potential(func):
    """defines the potential"""
    V = np.zeros(func.shape)
    for n, _ in np.ndenumerate(func):
        index_arr = np.array(n)
        V[n]=mu/8*(epsilon**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2
    return V


def laplace(func):
    """calculating the laplacian of ndarray"""
    shape = np.shape(func)
    D = len(shape)
    lap = -2*D*func
    for j in range(D):
        lap += (np.roll(func, -1, axis=j)
                +np.roll(func, 1, axis=j))
    return lap


def kinetic_hamilton(func):
    """calculating the free hamiltonian"""
    return -1/(2*mu*epsilon**2)*laplace(func)


def hamilton(func):
    """calculating the hamiltonian for double harmonic well"""
    return -1/(2*mu*epsilon**2)*laplace(func)+np.multiply(V,func)


def test_linearity(Hamiltonian,psi_in,iterations):
    shape = np.shape(psi_in)
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    err = []
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = Hamiltonian(alpha[i]*psi1 + beta[i]*psi2)
        RHS = alpha[i]*Hamiltonian(psi1) + beta[i]*Hamiltonian(psi2)
        error = np.max(np.abs(LHS - RHS))
        err.append(error)
    return err


def test_hermiticity(Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    err = []
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = inner_product(psi1, Hamiltonian(psi2))
        RHS = inner_product(Hamiltonian(psi1), psi2)
        error = np.abs(LHS - RHS)
        err.append(error)
    return err


def test_positivity(Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    count = 0
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        #print("Potential:"  ,np.sign(inner_product(psi1, np.multiply(potential(psi1),psi1)).real))
        #print("Hamiltonian:" , np.sign(inner_product(psi1, Hamiltonian(psi1)).real))
        if inner_product(psi1, Hamiltonian(psi1))<0:
            count +=1
    print("the hamiltonian has been negative " + str(count) + " out of " + str(iterations) + " times")
    return(count)


def test_eigenvectors(Kinetic_Hamiltonian, psi_in, iterations):
    shape = np.shape(psi_in)
    D = len(shape)
    plane_wave = np.zeros(shape, dtype=complex)
    err = []
    for i in range(iterations):
        k = np.random.randint(-N,N, size= D)
        eigenvalue = 0
        for index, value in np.ndenumerate(psi_in):
            plane_wave[index] = np.exp(2*np.pi * 1j * np.dot(np.array(index),k)/N)
        LHS = Kinetic_Hamiltonian(plane_wave)
        for i in range(len(k)):
            eigenvalue += (np.sin(np.pi/N * k[i]))**2
        RHS = 2/(mu*epsilon**2)*eigenvalue * plane_wave
        error = np.max(np.abs(LHS - RHS ))
        err.append(error)
    return err





def so_integrator(func,M):
    """solves the time dependent schrödinger equation for a given wavefunction with the second-order integrator"""
    start = func
    for m in np.arange(0,M):
        iteration = start - 1j*tau*hamilton(start) - 1/2*tau**2*hamilton(hamilton(start))
        start = iteration
    return iteration




def Exponential_potential(psi_in, time_step):
    shape = np.shape(psi_in)
    squares = np.zeros(shape, dtype=complex)
    for index, value in np.ndenumerate(psi_in):
        squares[index] = np.dot(np.array(index)-int(N/2) , np.array(index)-int(N/2))
    potential_term = mu/8 * (epsilon**2 * squares - 1)**2
    psi_out = np.exp(-1j*time_step/2 * potential_term)
    psi_out = np.multiply(psi_out, psi_in)
    return psi_out


def Exponential_kinetic(psi_in, time_step):
    shape = np.shape(psi_in)
    eigenvalues = np.zeros(shape)
    D = len(shape)
    for index, value in np.ndenumerate(psi_in):
        for i in range(D):
            eigenvalues[index] += 2/(mu*epsilon**2)*(np.sin(np.pi/N*index[i]))**2
    psi_out = np.multiply(np.exp(-1j*time_step * eigenvalues), psi_in)
    return psi_out


def Strang_Splitting(psi_in,M):
    for i in range(M):
        psi_out = Exponential_potential(psi_in, tau)
        psi_out = fft(psi_out)
        psi_out = Exponential_kinetic(psi_out, tau)
        psi_out = ifft(psi_out)
        psi_out = Exponential_potential(psi_out,tau)
        psi_in = psi_out
    return psi_out



def test_unitarity(integrator, psi_in, iterations):
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = normalize(phi)
    err = []
    for i in range(iterations):
        wave = integrator(phi,iterations)   
        norm = np.sqrt(inner_product(wave,wave))
        error = np.abs(1-norm)
        err.append(error)
        phi = wave
    return err

def test_linearity_integrator(integrator, psi_in, iterations):
    shape = np.shape(psi_in)
    alpha = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    beta = np.random.rand(iterations) + 1j * np.random.rand(iterations)
    err = []
    for i in range(iterations):
        psi1 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        psi2 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        LHS = integrator((alpha[i]*psi1 + beta[i]*psi2), 1)
        RHS = alpha[i]*integrator(psi1, 1) + beta[i]*integrator(psi2,1)
        error = np.max(np.abs(LHS - RHS))
        err.append(error)
    return err
       
    
def test_energy_conserv(integrator, psi_in, iterations): 
    shape = np.shape(psi_in)
    phi = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    phi = normalize(phi)
    energy0 = inner_product(phi, hamilton(phi))
    err = []
    for i in range(iterations):
        wave = integrator(phi, iterations) 
        energy0 = inner_product(phi, hamilton(phi))/inner_product(phi, phi)
        energy1 = inner_product(wave, hamilton(wave))/inner_product(wave, wave)
        error = np.abs(energy1 - energy0)
        err.append(error)
        phi = wave        
    return err



def images(func, integr):
    """ creates a list of the calculated wavefunctions for all timesteps """
    start = func
    ims = []
    ims.append(start)
    M_step = int(M/(FRAMES-1))       # dont save all the frames from the time evolution, just the ones we use                           
    for i in np.arange(1,FRAMES):
        iteration = integr(start,M_step)
        start = iteration
        ims.append(iteration)
    return ims


def rel():
    global M, tau
    M_save = M
    tau_save = tau
    Ms = []
    E_so = []
    E_st = []
    norm_so = []
    avg_diff = []
    for m in range(100)[1:]:
        M = 10*m
        tau = W*1/M # using T=1
        so = so_integrator(Psi, M)
        st = Strang_Splitting(Psi, M)
        Ms.append(M)
        E_so.append(inner_product(so, hamilton(so)))
        E_st.append(inner_product(st, hamilton(st)))
        norm_so.append(inner_product(so, so))
        avg_diff.append(np.average(np.abs(so-st)))
        print(str(M) + " out of 990")
    M = M_save
    tau = tau_save
    
    figure, axs = plt.subplots(2,2)
    axs[0,0].set(xlabel=r'log(M)')
    axs[0,1].set(xlabel=r'log(M)')
    axs[1,0].set(xlabel=r'log(M)')
    axs[1,1].set(xlabel=r'log(M)')
    axs[0,0].plot(np.log(Ms), np.log(np.array(E_so)/np.array(norm_so)), label=r'$log\left(\frac{\langle\hat{\Psi}_{so}|\hat{H}|\hat{\Psi}_{so}\rangle}{\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle}\right)$')  #,   
    axs[0,1].plot(np.log(Ms), np.log(E_st), label=r'$log(\langle\hat{\Psi}_{st}|\hat{H}|\hat{\Psi}_{st}\rangle)$')
    axs[1,0].plot(np.log(Ms), np.log(norm_so), label=r'$log(\langle\hat{\Psi}_{so}|\hat{\Psi}_{so}\rangle)$')
    axs[1,1].plot(np.log(Ms), np.log(avg_diff), label="log(avg($|\hat{\Psi}_{so}-\hat{\Psi}_{st}|$))")
    axs[0,0].legend(fontsize=18)
    axs[0,1].legend(fontsize=18)
    axs[1,0].legend(fontsize=18)
    axs[1,1].legend(fontsize=18)
    return Ms, E_so, E_st, norm_so, avg_diff

""" --- run the code --- """

n, Psi=gaussian_1D(-int(N/4),int(N/16))
V = potential(Psi)
Psi = normalize(Psi)


Ms, E_so, E_st, norm_so, avg_diff=rel()

fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(12, 6))

line1, = ax1.plot([], [], label=r'$|\hat{\Psi}_{so}|^2\cdot\frac{1}{\varepsilon}$')  
line2, = ax2.plot([], [], label=r'$|\hat{\Psi}_{st}|^2\cdot\frac{1}{\varepsilon}$')
line3, = ax3.plot([], [], label=r'$|\hat{\Psi}_{so}-\hat{\Psi}_{st}|^2$')  

ax12 = ax1.twinx()
ax12.plot(n*epsilon,V/(H_BAR*W),color="C1", label=r'$\frac{V}{\hbar\omega}$')
ax22 = ax2.twinx()
ax22.plot(n*epsilon,V/(H_BAR*W),color="C1", label=r'$\frac{V}{\hbar\omega}$')


ax1.set(xlim=[-int(N/2)*epsilon,int(N/2)*epsilon], ylim=[0,4], 
        xlabel=r'$\frac{x}{r}$', title='Second-order integrator')
ax1.tick_params(axis='y', labelcolor="C0")
ax12.set(xlim=[-int(N/2)*epsilon,int(N/2)*epsilon], ylim=[0,60])
ax12.tick_params(axis='y', labelcolor="C1")
ax2.set(xlim=[-int(N/2)*epsilon,int(N/2)*epsilon], ylim=[0,4], 
        xlabel=r'$\frac{x}{r}$', title='Strang-splitting integrator')
ax2.tick_params(axis='y', labelcolor="C0")
ax22.set(xlim=[-int(N/2)*epsilon,int(N/2)*epsilon], ylim=[0,60])
ax22.tick_params(axis='y', labelcolor="C1")
ax3.set(xlim=[-int(N/2)*epsilon,int(N/2)*epsilon], ylim=[0,10**(-5)], 
        xlabel=r'$\frac{x}{r}$', title='Difference between both integrators')


fig.suptitle(r'$\mu$={0}, $\varepsilon$={1}, N={2}, M={3}, T={4}, $\tau$={5}'.format(mu,round(epsilon, 5),N,M,T,tau), fontsize=12)

ax1.legend(loc=2)
ax12.legend(loc=1)
ax2.legend(loc=2)
ax22.legend(loc=1)
ax3.legend(loc=2)


def animate(y,line):
    line.set_data(n*epsilon,y)

def animate_all(i):  
    animate(abs(images_so[i])**2/epsilon , line1)      # second-order
    animate(abs(images_strang[i])**2/epsilon , line2)      # strang-splitting
    animate(abs(images_strang[i]-images_so[i])**2 , line3)   # difference between both
    return line1, line2, line3,




iterations = 10
##### all the tests - with naming of them in the print()
print("testing linearity of the hamiltonian. Maximum error: " + str(np.max(np.abs(test_linearity(hamilton,Psi,iterations)))))
print("testing hermicity of the hamiltonian. Maximum error: " + str(np.max(np.abs(test_hermiticity(hamilton, Psi, iterations)))))
print("testing positivity of the hamiltonian.")
test_positivity(hamilton, Psi, iterations)
print("testing eigenvectors of the kinetic hamiltonian. Maximum error: " + str(np.max(np.abs(test_eigenvectors(kinetic_hamilton, Psi, iterations)))))
print("testing unitarity of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_unitarity(so_integrator, Psi, iterations)))))
print("testing unitarity of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_unitarity(Strang_Splitting, Psi, iterations)))))
print("testing linearity of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_linearity_integrator(so_integrator, Psi, iterations)))))
print("testing linearity of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_linearity_integrator(Strang_Splitting, Psi, iterations)))))
print("testing energy conservation of the Second-Order integrator. Maximum error: " + str(np.max(np.abs(test_energy_conserv(so_integrator, Psi, iterations)))))
print("testing energy conservation of the Strang-Splitting integrator. Maximum error: " + str(np.max(np.abs(test_energy_conserv(Strang_Splitting, Psi, iterations)))))



images_so = images(Psi, so_integrator) 
images_strang = images(Psi, Strang_Splitting)

anim = animation.FuncAnimation(fig, animate_all, frames = FRAMES, interval = 1000/FPS, blit = True) 

#anim.save('animation_project.gif', writer = 'pillow', fps = FPS)     # to save the animation


plt.show()
