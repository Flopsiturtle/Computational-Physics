#Florian Telleis
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
import matplotlib.animation as animation




""" --- constants and parameters --- """

"""lattice"""
# D = len(wave.shape) 
N = 200  # number of points in each direction of D-dimensional lattice  # if wavefunction is given: N = (wave.shape)[0]    # \\ [0] is arbitrary because quadratic matrix
A = 0.3  # spacing between lattice points   # assumption: A is input | can also be variant of L=N*a  
"""potential/hamiltonian"""
R = 18  # length from zero-point to potential-valleys 
M = 0.1   # mass of point particle
W = 5   # frequency
H_BAR = 1#!!! actually: 6.62607015*10**(-34)    # J*s
mu = (M*W*R**2)/H_BAR
epsilon = A/R
"""integrator"""
M = 10000   # large value
T = 10      # time
tau = T/M   # time step
"""animation"""
FRAMES = 200    # number of frames in final animation
FPS = int(FRAMES/T)     # number of frames per second if given time T is real time in seconds
#FPS = 23    # if FPS should be indipendent of T




''' --- wavefunction --- '''

def gaussian_1D(mean,sigma): 
    x_data = np.arange(0, N) 
    y_data = stats.norm.pdf(x_data, mean, sigma) 
    return y_data 

def wave_to_lattice(wave):  
    """transform given wavefunction "wave" to lattice units (discrete)"""
    D = len(wave.shape)             
    return A**(D/2)*wave  

test_function = wave_to_lattice(gaussian_1D(25,10)) # choosing the wavefunction for the animation




""" --- hamiltonian and CO. --- """

def laplace_in_lattice(phi):    # input phi has to be in lattice units
    """calculate the laplacian of a wavefunction "phi" """
    shape = phi.shape
    D = len(shape)      
    laplace = np.zeros(shape)
    for i in np.arange(0,D):
        laplace = laplace + np.roll(phi,1,i) - 2*phi + np.roll(phi,-1,i)
    return laplace


def potential_in_lattice(phi):
    """define the potential in lattice units with the same dimensions as input wavefunction "phi" """
    V = np.zeros(phi.shape)
    for index, value in np.ndenumerate(phi):
        index_arr = np.array(index)
        V[index] = mu/8 * (epsilon**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2       
    return V        


def hamiltonian_in_lattice(phi):
    """calculate the hamiltonian of a given wavefunction "phi" """    
    H = - 1/(2*mu*epsilon**2)*laplace_in_lattice(phi) + potential_in_lattice(phi) * phi
    return H


def scal_prod(phi1,phi2):
    """calculates the scalar product/inner product of two wavefunctions"""
    return np.sum(np.multiply(np.conjugate(phi1),phi2))




### Mickey
"""check for characteristics: - linear, - hermitian, -""" 
### Mickey







""" --- second order integrator --- """

def so_integrator(phi0):
    """solves the time dependent schrödinger equation for a given wavefunction "phi0" with the second-order integrator"""
    start = phi0
    for m in np.arange(0,M):
        iteration = start - 1j*tau*hamiltonian_in_lattice(start) - 1/2*tau**2*hamiltonian_in_lattice(hamiltonian_in_lattice(start))
        start = iteration
    return iteration



""" --- check unitarity --- """

def check_so_unitarity_explicitly(phi0):   
    """checks the unitarity of the second-order evolution operator via conservation of the norm of states"""   
    phitau = so_integrator(phi0)
    return print('The difference between the two norms is:', abs((scal_prod(phi0,phi0).real)-(scal_prod(phitau,phitau).real)))




""" --- check error --- """
######### ???????




""" --- create list of time evolution --- """ 

def so_integrator_images(phi0):
    """ creates a list of the calculated wavefunctions for all timesteps of the second-order time evolution """
    start = phi0
    ims = []
    ims.append(start)
    for m in np.arange(0,M):
        iteration = start - 1j*tau*hamiltonian_in_lattice(start) - 1/2*tau**2*hamiltonian_in_lattice(hamiltonian_in_lattice(start))
        start = iteration
        ims.append(iteration)
    return ims

images = so_integrator_images(test_function)    # creates the list of arrays for our test function
last = len(images)-1



#############
""" now simpler calculating of so_unitarity"""
def check_so_unitarity_images(phi0):   
    """checks the unitarity of the second-order evolution operator via conservation of the norm of states"""   
    phitau = images[last]
    return print('The difference between the two norms is:', abs((scal_prod(phi0,phi0).real)-(scal_prod(phitau,phitau).real)))
##############



""" --- animate the second order time evolution --- """
fig = plt.figure()
axis = plt.axes(xlim=(0,N-1),ylim =(0, 0.002))  # for constant axis
#fig, axis = plt.subplots()     # for variable axis

line, = axis.plot([], [])  

def init():  
    line.set_data([], []) 
    return line, 
   
def animate_so_integrator(i): 
    """function that gets called for animation"""
    """takes a number of arrays from calculated "images" corresponding to FRAMES and sets a line data"""
    global FRAMES,images,last
    i2 = int(i*last/(FRAMES-1)) # dont use all the frames from the time evolution
    y = abs(images[i2])**2
    x = np.arange(0,len(y)) 
    #axis.set_ylim(0, max(y)*1.1) # for variable axis
    line.set_data(x, y) 
    return line, 

anim = animation.FuncAnimation(fig, animate_so_integrator, init_func = init, frames = FRAMES, interval = 1000/FPS, blit = True) 

anim.save('animation_so_integrator_3-blitTrue.gif', writer = 'pillow', fps = FPS) 




''' plot of the last frame of the animation with potential '''

###########?????? how do we show the potential in regards to abs(wave)**2 ??????


y_values1 = abs(images[last])**2
x_values1 = np.arange(0,len(y_values1))

y_values3 = potential_in_lattice(test_function)
x_values3 = np.arange(0,len(y_values3))

fig=plt.figure()
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

ax.plot(x_values1, y_values1, color="C0")
ax.set_xlabel("x label 1", color="C0")
ax.set_ylabel("y label 1", color="C0")
ax.tick_params(axis='x', colors="C0")
ax.tick_params(axis='y', colors="C0")

ax2.plot(x_values3, y_values3, color="C1")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('x label 3', color="C1") 
ax2.set_ylabel('y label 3', color="C1")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="C1")
ax2.tick_params(axis='y', colors="C1")



plt.show()







'''2D'''
#f = potential_in_lattice(test_function)
#g = hamiltonian_in_lattice(test_function).real
#h = abs(so_integrator(test_function))

#plt.imshow(f, interpolation='none')
#plt.title('potential: A={0}, R={1}, size={2}'.format(A,R,size))
#plt.show()
#plt.imshow(abs(test_function), interpolation='none')
#plt.title('abs-wavefunction: A={0}, R={1}, size={2}'.format(A,R,size))
#plt.show()
#plt.imshow(h, interpolation='none')
#plt.title('abs-so_integrator: A={0}, R={1}, size={2}, M={3}, T={4}'.format(A,R,size,M,T))
#plt.show()