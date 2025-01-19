import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace

import time

time_start = time.perf_counter()


### create and save test arrays
#np.random.seed(1)
#test_probs = np.random.rand(2,500)  # 2D array with 500 elements: 1D is values and 2D is errors - for 500 replicas
#np.random.seed(20)
#test_probs2 = np.random.rand(2,500)
#np.savetxt('test_probs2.txt',test_probs2)

### initiate which array
probs_array = np.loadtxt('prob_distr results/test_probs2.txt')




############## how do we want to incorporate errors into distribution?
######### maybe distribution of errors (easy implementation) - but then no meaning of which values have which errors




''' --- easiest way for a histogram --- '''
data = probs_array[0]
num_bars = 50
plt.hist(data, density=False, bins=num_bars, color='green')         # , histtype='step'
#plt.xticks(range(num_bars+1)) ---- wrong - would have to be with max/min

print(time.perf_counter() - time_start)     # 0.12-0.16
plt.show()




''' --- my own version --- '''
# ----- does the same
##### pro for mine: can bd changed to our meaning for error implementation
##### con for minr: maybe slower, does not easily change between histo and density (fix: number/probs.shape)
def prob_distr(probs_array,num_bars):
    probs, probs_err = probs_array
    max = np.max(probs)
    min = np.min(probs)
    bar_size = (max-min)/num_bars
    bars = []
    for i in np.arange(num_bars):
        number = probs[(probs>=(min+bar_size*i)) & (probs<(max+bar_size*(i+1-num_bars)))]
        bars.append(number.size)
    bars[num_bars-1] += 1
    bars.append(0)
    x = np.arange(min,max+bar_size,bar_size)
    return x,bars,bar_size
    #### below was for different code with error
    if np.sum(bars) == probs.shape:
        return x,bars,bar_size
    else:
        return print('!!!wrong count in histogramm!!!')



num_bars = 50
x,y,width = prob_distr(probs_array,num_bars)
plt.bar(x,y,width,align='edge', color='C0')
#plt.xticks(x)


print(time.perf_counter() - time_start)     # nimmt sich eigentlich nichts -> falls wir in meiner Funktion mehr Freiheit / die errors besser einbauen kann dann meine
plt.show()                                   # ansonsten die Standard library zur Sicherheit




''' --- distributiuon using gaussian kernel --- '''
####### cool but problem: wants to be zero at the edges! - could work for our values because gaussian i think      (even more at right because of the zero at end)
data = probs_array[0]
kde = gaussian_kde(data)
dist_space = linspace(min(data),max(data),100)
plt.plot(dist_space,(kde(dist_space)/np.max(data))*(np.max(y)), color='red')

plt.show()


