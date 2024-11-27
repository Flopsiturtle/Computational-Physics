import numpy as np
from scipy import stats 

N = 200
mu = 153.9
epsilon = 1/60
M = 10000
tau = 10/M



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