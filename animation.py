import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import variables
from variables import *   
import hamiltonian
import integrators


FRAMES = 200    
FPS = int(FRAMES/10)    # animation is 10 seconds long
#FPS = 23   # variable FPS


def images(func, integr):
    """ creates a list of the calculated wavefunctions for all timesteps """
    start = func
    ims = []
    ims.append(start)
    M_step = int(M/(FRAMES-1))       # dont save all the frames from the time evolution, just the ones we use                           
    for i in np.arange(1,FRAMES):
        iteration = integr(start,M_step,tau)
        start = iteration
        ims.append(iteration)
    return ims


def animate(y,line):
    line.set_data(n*epsilon,y)

def animate_all(i):  
    animate(abs(images_so[i])**2/epsilon , line1)      # second-order
    animate(abs(images_strang[i])**2/epsilon , line2)      # strang-splitting
    animate(abs(images_strang[i]-images_so[i])**2 , line3)   # difference between both
    return line1, line2, line3,


''' use wavefunction '''

n, Psi = variables.gaussian_1D(-int(N/4),int(N/16))
V = hamiltonian.potential(Psi)
Psi = variables.normalize(Psi)


''' create animation '''

fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(12, 6))

line1, = ax1.plot([], [], label=r'$|\hat{\Psi}_{so}|^2\cdot\frac{1}{\varepsilon}$')  
line2, = ax2.plot([], [], label=r'$|\hat{\Psi}_{st}|^2\cdot\frac{1}{\varepsilon}$')
line3, = ax3.plot([], [], label=r'$|\hat{\Psi}_{so}-\hat{\Psi}_{st}|^2$')  

ax12 = ax1.twinx()
ax12.plot(n*epsilon,V,color="C1", label=r'$\frac{V}{\hbar\omega}$')   ### hbar omega!?
ax22 = ax2.twinx()
ax22.plot(n*epsilon,V,color="C1", label=r'$\frac{V}{\hbar\omega}$')   ### hbar omega!?


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


fig.suptitle(r'$\mu$={0}, $\varepsilon$={1}, N={2}, M={3}, $\tau$={4}'.format(mu,round(epsilon, 5),N,M,tau), fontsize=12)

ax1.legend(loc=2)
ax12.legend(loc=1)
ax2.legend(loc=2)
ax22.legend(loc=1)
ax3.legend(loc=2)



images_so = images(Psi, integrators.so_integrator) 
images_strang = images(Psi, integrators.Strang_Splitting)

anim = animation.FuncAnimation(fig, animate_all, frames = FRAMES, interval = 1000/FPS, blit = True) 

#anim.save('animation_project_modular.gif', writer = 'pillow', fps = FPS)     # to save the animation


plt.show()