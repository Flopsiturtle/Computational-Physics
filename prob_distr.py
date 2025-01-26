import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.stats.kde import gaussian_kde
from numpy import linspace




''' --- implementing bootstrap-method for our 500 replicas --- '''

def bootstrap_samples(data,numb_samples,size_small_sample):
    boot_samples = []
    for i in np.arange(numb_samples):
        samples = []
        for i in np.arange(len(data)/size_small_sample):
            samp = random.sample(data.tolist(), size_small_sample)      # take random elements of given data, in total 500
            samples.append(samp)
        samples_list = np.concatenate(samples)    # and put all 500 into one sample array, working like a new array of 500 replicas
        boot_samples.append(samples_list)
    return boot_samples

def mean_error_hist(boot_samples,num_bars):
    boot_histos = []
    for j in np.arange(len(boot_samples)):
        y,binEdges = np.histogram(boot_samples[j],bins=num_bars)   # create histogramm for all samples (each size 500) of boot-method
        boot_histos.append(y)
    mean_histo_ordered = []
    err_histo_ordered = []
    for i in np.arange(num_bars):
        mean_samples = [item[i] for item in boot_histos]     # take only specific bar elements of each sample
        mean_mean = np.mean(mean_samples)
        R = len(mean_samples)
        err_mean = np.sqrt(np.sum((mean_samples-mean_mean)**2)/(R*(R-1)))
        mean_histo_ordered.append(mean_mean)
        err_histo_ordered.append(err_mean)
    return mean_histo_ordered,err_histo_ordered,binEdges



''' --- function for plotting the final histogram  --- '''

def final_histogramm(TYPE,data,calc_mean_mean,num_bars,numb_samples,size_small_sample):
    ### data
    boot_samples = bootstrap_samples(data,numb_samples,size_small_sample)
    data_boots,error_boots,binEdges = mean_error_hist(boot_samples,num_bars)
    bar_centers = 0.5*(binEdges[1:]+binEdges[:-1])
    bar_size = ((np.max(binEdges)-np.min(binEdges))/num_bars)
    ### plotting
    plt.figure(figsize=(9,6))
    plt.bar(bar_centers, data_boots, width=bar_size, color=['cornflowerblue','royalblue'], label="bars")    # histogram from bootstrap
    plt.errorbar(bar_centers, data_boots, fmt=" ", yerr=error_boots, color='black', capsize=2, label="error")      # error-bars from bootstrap
    plt.axvline(x=calc_mean_mean, color='red', label="mean-mean calc")       # visaulizing the bootstrap calculated mean
    plt.axvline(x=np.mean(boot_samples), linestyle='dashed', color='limegreen', label="mean-mean boot")       # visaulizing the beforehand calculated mean
    plt.legend(loc="upper left")
    plt.xlabel('value')
    plt.ylabel('counts')
    if TYPE == 0:
        plt.title('Magnetizazion: #bars = {0}, #boot samples = {1}, size small samples = {2}'.format(num_bars,numb_samples,size_small_sample))
    if TYPE == 1:
        plt.title('Energy: #bars = {0}, #boot samples = {1}, size small samples = {2}'.format(num_bars,numb_samples,size_small_sample))
    plt.show()


''' --- run the code --- '''
### initiate which array
probs_array_magn = np.loadtxt('prob_distr results/test_probs_gauss2.txt') # change for magn
data_magn = probs_array_magn[0]
probs_array_energy = np.loadtxt('prob_distr results/test_probs_gauss2.txt') # change for energy
data_energy = probs_array_energy[0]
### define what is the beforehand calculated mean
calc_mean_mean_magn = 10
calc_mean_mean_energy = 10

### set number of bars for histograms
num_bars = 50
### set parameters for bootstrap-method
numb_samples = 50
size_small_sample = 5

# magnetization
final_histogramm(0,data_magn,calc_mean_mean_magn,num_bars,numb_samples,size_small_sample)
# energy
final_histogramm(1,data_energy,calc_mean_mean_energy,num_bars,numb_samples,size_small_sample)

###### titles have to be different!!!!!



exit()

''' --- distributiuon using gaussian kernel --- '''
y,binEdges = np.histogram(data,bins=num_bars)
kde = gaussian_kde(data)
dist_space = linspace(min(data),max(data),100)
plt.plot(dist_space,kde(dist_space)/np.max(kde(dist_space))*(np.max(y)), color='purple')
plt.show()



exit()

''' --- for checking: histogram using original replicas without bootsrap --- '''
bar_centers = 0.5*(binEdges[1:]+binEdges[:-1])
bar_size = ((np.max(binEdges)-np.min(binEdges))/num_bars)
plt.bar(bar_centers,y, width=bar_size, color='green')         # , histtype='step'

plt.show()














exit()





### create and save test arrays
#np.random.seed(1)
#test_probs = np.random.normal(10,1,(2,500)) # mean 10 with st_dev 1
#test_probs[1]=(np.random.rand(1,500))   # errors not gaussian
#np.random.seed(20)
#test_probs2 = np.random.rand(2,500)
#np.savetxt('test_probs_gauss.txt',test_probs)


#plt.bar(bincenters, y, width=bar_size, color='r', yerr=menStd)
#plt.show()


############## how do we want to incorporate errors into distribution?
######### maybe distribution of errors (easy implementation) - but then no meaning of which values have which errors
#----- he says try bootstrap method or doing way more replicas and then statistical mean (same as before from lecture) over means -> use overall mean as value for histogramm and error of value as error for histrogramm



''' --- easiest way for a histogram --- '''
#weights = np.ones_like(data)/float(len(data))   # add weights=weights into plt.hist() for density plot
#plt.hist(data,density=False, bins=num_bars, color='green')         # , histtype='step'
#plt.xticks(.....) #would have to be with max/min -> my function
#plt.show()




#plt.bar(bar_centers, data_boots, width=bar_size, yerr=error_boots,facecolor='k',alpha=0.1)
#plt.step(bar_centers, data_boots,'k',linestyle='--',linewidth=1)



''' --- my own version --- '''
# ----- does the same
##### pro for mine: can be changed to our meaning, maybe for error implementation
##### con for mine: maybe slower, does not easily change between histo and density (fix: number/probs.shape)
def prob_distr(probs_array,num_bars,density):
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
    if density == 0:
        for i in np.arange(num_bars):
            bars[i] = bars[i]/((probs.shape)[0])
        return x,bars,bar_size
    return x,bars,bar_size




#x,y,width = prob_distr(probs_array,num_bars,density=1)
#plt.bar(x,y,width,align='edge', color='C0')
#plt.xticks(x)

#plt.show()                                   


#exit()
