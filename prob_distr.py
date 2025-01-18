import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace


### create and save test arrays
#np.random.seed(1)
#test_probs = np.random.rand(2,500)  # 2D array with 500 elements: 1D is values and 2D is errors - for 500 replicas
#np.random.seed(20)
#test_probs2 = np.random.rand(2,500)
#np.savetxt('test_probs2.txt',test_probs2)



def prob_distr(probs_array,num_bars):
    probs, probs_err = probs_array
    max = np.max(probs)
    min = np.min(probs)
    bar_size = (max-min)/num_bars
    bars = []
    for i in np.arange(num_bars):
        number = probs[(probs>=(min+bar_size*i)) & (probs<(max+0.0001+bar_size*(i+1-num_bars)))]
        bars.append(number.size)
    bars.append(0)
    x = np.arange(min,max+bar_size,bar_size)
    return x,bars,bar_size



### initiate which array
probs_array = np.loadtxt('test_probs.txt')


### my distribution function
num_bars = 50
x,y,width = prob_distr(probs_array,num_bars)
print(x)
print(y)
plt.bar(x,y,width,align='edge')
#plt.xticks(x)


### distributiuon using gaussian kernel
##### cool but problem: wants to be zero at the edges!      (even more at right because of the zero at end)
data = probs_array[0]
kde = gaussian_kde(data)
dist_space = linspace(min(data),max(data),100)
plt.plot(dist_space,(kde(dist_space)/np.max(data))*(np.max(y)), color='red')



plt.show()