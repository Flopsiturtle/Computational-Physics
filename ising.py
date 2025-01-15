
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1)

N = 200
beta = 1
b = 0
config = 100

spin_grid = np.random.choice([-1,1],size=(N,N))
shape = np.shape(spin_grid)
dim = len(shape)

def hamilton_diff(spin_grid, index):
    value = 0
    summ = 0
    for i in range(dim):
        summ += np.roll(spin_grid, -1, axis = i)[index] + np.roll(spin_grid, +1, axis = i)[index]
    value += 2*spin_grid[index]*(beta*summ + b)
    return value
    

def metropolis(spin_grid):
    for index, s in np.ndenumerate(spin_grid):
        exp_diff = np.exp(-hamilton_diff(spin_grid, index))
        r = np.random.rand()
        if exp_diff > r :
            spin_grid[index] = - spin_grid[index]
    return spin_grid


def magnet(spin_grid):
    value = 0
    for index, s in np.ndenumerate(spin_grid):
        value += s
    return value/(N**dim)

def markov_chain(initial_spins, iterations):
    m = []
    m.append(magnet(initial_spins))
    for i in range(iterations):
        result = metropolis(initial_spins)
        m.append(magnet(result))
        initial_spins = result
    return m

x = range(config + 1)
plt.scatter(x, markov_chain(spin_grid, config), s=1)


