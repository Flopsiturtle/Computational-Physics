#Florian Telleis
import numpy as np
# import cmath
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

""" --- constans and parameters --- """

# D = len(wave.shape)
# N = (wave.shape)[0]
A = 0.3  # for viewing good 0.13             # assumption: input is a | can also be variant of L=N*a with N=(wave.shape)[0]  \\ [0] is arbitrary because quadratic matrix


R = 36  # length from zero-point to potential-valleys 
M = 1   # mass of point particle
W = 1   # ... in units of time^-1
H_BAR =1#!!! actually: 6.62607015*10**(-34)    # J*s


mu = (M*W*R**2)/H_BAR
epsilon = A/R




""" --- hamiltonian and CO. --- """


def wave_to_lattice(wave):   # transformation can also be done implicitly in a later function 
    """transform given wavefunction "wave" to lattice units (discrete)"""
    D = len(wave.shape)             
    return A**(D/2)*wave        

# phi = wave_to_lattice(wave,a)


def laplace_in_lattice(phi):    # input phi has to be in lattice units
    """calculate the laplacian of a wavefunction "phi" """
    shape = phi.shape
    D = len(shape)      # assumption: quadratic arrays; for example 2D: NxN
    laplace = np.zeros(shape)
    for i in np.arange(0,D):
        laplace = laplace + np.roll(phi,1,i) - 2*phi + np.roll(phi,-1,i)
    return laplace


def potential_in_lattice(phi):
    """define the potential in lattice units with the same dimensions as input wavefunction "phi" """
    V = np.zeros(phi.shape)
    N = phi.shape[0]
    for index, value in np.ndenumerate(phi):
        index_arr = np.array(index)
        V[index] = mu/8 * (epsilon**2*np.dot(index_arr-int(N/2),index_arr-int(N/2))-1)**2       
    return V        
                                # move potential to lattice centrum to obtain all wanted values! -> index_arr-int(n/2)
                                # i understood the lattice moving for the potential
                            ##### BUT how does this influence in which way we input our given wavefunction??? is it already central????

def hamiltonian_in_lattice(phi):
    """calculate the hamiltonian of a given wavefunction "phi" """    
    #phi = wave_to_lattice(wave) # phi already input
    H = - 1/(2*mu*epsilon**2)*laplace_in_lattice(phi) + potential_in_lattice(phi) * phi
    return H



# scalar product
def scal_prod(phi1,phi2):
    """calculates the scalar product/inner product of two wavefunctions"""
    return np.sum(np.multiply(np.conjugate(phi1),phi2))





"""check for characteristics: - linear, - hermitian, -""" 
#!#!#!#!#! checks were done before upload of scipt with the info

# wrong
def check_hermitian(hamiltonian,error):      # geht nur für 2D arrays durch np.matrix!!!
    ham_matrix = np.matrix(hamiltonian)
    ham_adj = ham_matrix.getH()
        ## dont check hermitian of hamiltonian on phi but hamiltonian itself!
    if abs((ham_matrix - ham_adj)).all() <= error:
        result = "hermitian"
    else:
        result = "non-hermitian"
    return print(result)
# wrong

def check_linear(wave,error):
                # hamilton(lambda*phi) == lambda*hamilton(phi)      # -> welche Zahl als Beispiel?: einfach mal 27.9
                # hamilton(phi+phi´) == hamilton(phi) + hamilton(phi´)      # wie phi´ aus phi erzeugen?: wave@wave
    wave2 = wave@wave
    linear1 = hamiltonian_in_lattice(27.9*(wave-wave2))
    linear2 = 27.9*(hamiltonian_in_lattice(wave)-hamiltonian_in_lattice(wave2))
    if (abs(linear1 == linear2)).all() <= error:
        result = "linear"
    else:
        result = "non-linear"
    return print(result)






""" --- integrators --- """

M = 5000     # large value
T = 7   # time
tau = T/M   # time step


def so_integrator(phi0):
    """solves the time dependent schrödinger equation for a given wavefunction "phi0" with the second-order integrator"""
    start = phi0
    for m in np.arange(1,M+1):
        iteration = start - 1j*tau*hamiltonian_in_lattice(start) - 1/2*tau**2*hamiltonian_in_lattice(hamiltonian_in_lattice(start))
        start = iteration
    return iteration
    # I think this is correctly implemented



""" --- check unitarity --- """

def check_so_unitarity(phi0,error):   
    """checks the unitarity of the second-order evolution operator via conservation of the norm of states"""   
    phitau = so_integrator(phi0)
    if abs((scal_prod(phi0,phi0).real)-(scal_prod(phitau,phitau).real)) <= error:
        result = "WARNING, evolution operator is unitary."
    else:
        result = "OK, evolution operator is not unitary."
    return print(result)

#############
##### !!! dont do the function with error input BUT print out the difference between the two scal_prod and from that output we manually evaluate the result for multiple conditions !!! 
#############




""" --- check error --- """

# wrong
def exp_array(array,k):
    array = np.identity() + array
    for i in np.arange(0,k+1):
        array = (array**k)/math.factorial(k)    ### array**k is different than wanted definition 
    return array
# wrong








# ----- tests -----





test_2D = np.array([[[1.3,4.27],[3.6,5.9]]])
test_2D2 = np.array([[[2,5+1j],[3,9+4j]]])
test_3D = np.array([[[2,32],[2,1]],[[2,7],[2,3]]])


size = 200
test_2D3 = np.random.rand(size,size) + 1j * np.random.rand(size,size)
#test_2D3 = np.random.rand(size,size)*0+1 + 1j * np.random.rand(size,size)*0 +1j

#test_1D = np.random.rand(size,1)+ 1j * np.random.rand(size,1)
test_1D = np.random.rand(size,1)*0+1+ 1j * np.random.rand(size,1)*0 +1j


gaussian_1D = 1




#print(test_1D)

#print(scal_prod(test_2D3,test_2D3).real)

#ff = so_integrator(test_2D3)
#gg = so_integrator(ff)
#check_so_unitarity(gg,0.0001)



###################
''' !!! WAVEFUNCTION INPUT: choose initial condition so that wavefunction travels from left to right and you see a reflecting and transmitting part in the animated time evolution !!! -> could be gaussian which has peak at left curve'''
###################

''' 1D '''






''' animate so '''












##### animation via for loop

fy = potential_in_lattice(test_1D)
fx = np.arange(0,len(fy))
g = abs(hamiltonian_in_lattice(test_1D))**2
hy = abs(so_integrator(test_1D))**2
hx = np.arange(0,len(hy))




fig, ax = plt.subplots()
ax = plt.axes(xlim =(0, 200), ylim =(0, 5))  


def so_integrator_images(phi0):
    start = phi0
    ims = []
    ims.append(start)
    for m in np.arange(1,M+1):
        iteration = start - 1j*tau*hamiltonian_in_lattice(start) - 1/2*tau**2*hamiltonian_in_lattice(hamiltonian_in_lattice(start))
        start = iteration
        ims.append(iteration)
    return ims


images_1D = so_integrator_images(test_1D)

###########################

#fig, axis = plt.subplots()  

fig = plt.figure()
axis = plt.axes(xlim =(0, 200), ylim =(0, 5))  
  
# initializing a line variable 
line, = axis.plot([], [], lw = 3)  
   
# data which the line will  
# contain (x, y) 
def init():  
    line.set_data([], []) 
    return line, 
   
FRAMES = 100
FPS = 23
images = images_1D

def animate_so_integrator(i): 
    global FRAMES,FPS,images
    images = so_integrator_images(test_1D)
    i2 = i*int(len(images)/FRAMES)

    y = abs(images[i2])**2
    x = np.arange(0,len(y)) 

    line.set_data(x, y) 
    return line, 

anim = animation.FuncAnimation(fig, animate_so_integrator, init_func = init, frames = 200, interval = 1/FPS, blit = True) 
  
   
#anim.save('continuousSineWave.gif', writer = 'ffmpeg', fps = 30) 



###########################





x = hx
y = images_1D
 

# enable interactive mode
plt.ion()
 
# creating subplot and figure
fig = plt.figure()
#ax = fig.add_subplot(111)
ax = plt.axes(xlim =(0, 200), ylim =(0, 10))  
line1, = ax.plot(x, abs(y[0])**2)


FRAMES = 100
FPS = 23

for i in np.arange(0,len(images_1D),int(len(images_1D)/FRAMES)):
    line1.set_xdata(x)
    line1.set_ydata(abs(y[i])**2)
 
    # re-drawing the figure
    fig.canvas.draw()
     
    # to flush the GUI events
    fig.canvas.flush_events()
    time.sleep(1/FPS)




##### slower animation but axis are changing

FRAMES = 100
FPS = 23

for i in np.arange(0,len(images_1D),int(len(images_1D)/FRAMES)):
    ax.clear()
    ax.plot(x,abs(y[i])**2)
    # re-drawing the figure
    fig.canvas.draw()
     
    # to flush the GUI events
    fig.canvas.flush_events()
    time.sleep(1/FPS)







#ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)










''' plot der beiden '''



x_values1=fx
y_values1=fy

x_values2=np.array([0,0])
y_values2=np.array([0,0])

x_values3=hx
y_values3=hy


fig=plt.figure()
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
ax3=fig.add_subplot(111, label="3", frame_on=False)

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






######### random

#plt.plot(fx,fy)
#plt.title('potential: A={0}, R={1}, size={2}'.format(A,R,size))
#plt.show()

#plt.plot(np.arange(len(abs(test_1D))),abs(test_1D))
#plt.title('abs-wavefunction: A={0}, R={1}, size={2}'.format(A,R,size))
#plt.show()

#plt.plot(hx,hy)
#plt.title('abs-so_integrator: A={0}, R={1}, size={2}, M={3}, T={4}'.format(A,R,size,M,T))
#plt.show()


check_so_unitarity(test_1D,0.001)







'''2D'''


#f = potential_in_lattice(test_2D3)
#g = hamiltonian_in_lattice(test_2D3).real
#h = abs(so_integrator(test_2D3))

#plt.imshow(f, interpolation='none')
#plt.title('potential: A={0}, R={1}, size={2}'.format(A,R,size))
#plt.show()
#plt.imshow(abs(test_2D3), interpolation='none')
#plt.title('abs-wavefunction: A={0}, R={1}, size={2}'.format(A,R,size))
#plt.show()
#plt.imshow(h, interpolation='none')
#plt.title('abs-so_integrator: A={0}, R={1}, size={2}, M={3}, T={4}'.format(A,R,size,M,T))
#plt.show()





#print(laplace_in_lattice(test_2D3))
#check_hermitian(test_2D3,1)
#check_linear(test_2D3,0.01)


#print(so_integrator(test_2D3))
#a=so_integrator(test_2D3)

#check_so_unitarity(test_2D3,0.001)






#print(test_2D3)


#print(exp_array(test_2D,5))

#print(round(scal_prod(test_2D,test_2D).real,7))

#print(round(scal_prod(so_integrator(test_2D),so_integrator(test_2D)).real,8))


#print(check_hermitian(test_3D)) # geht nicht durch 3D

#print(check_hermitian(test_2D))
#print(check_hermitian(test_2D2))

#print(check_linear(test_2D,......))        inputs fehlen für hermitian


#print(np.arange(1,10+1))


#print(test)
#print(np.roll(test,1,0))

# axis 2 -> x
# axis 1 -> y
# axis 0 -> z




#input()