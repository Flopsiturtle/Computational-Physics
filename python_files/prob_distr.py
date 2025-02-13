import numpy as np
import matplotlib.pyplot as plt
import random
boot_seed = 5
random.seed(boot_seed) 

### only used for gaussian representation, not important final histograms
from scipy.stats.kde import gaussian_kde
from numpy import linspace



''' --- implementing bootstrap-method for our 500 replicas --- '''

def bootstrap_samples(data,numb_samples,size_small_sample):
    boot_samples = [data]   ###### take original data into account for later histogram means = use as one more sample replica
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
        y,binEdges = np.histogram(boot_samples[j],bins=num_bars)   # create histogram for all samples (each size 500) of boot-method
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
    mean_boot_samples = np.mean(boot_samples)
    R = len(boot_samples)
    err_mean_boot = np.sqrt(np.sum((boot_samples-mean_boot_samples)**2)/(R*(R-1)))     # statistical error of bootstrap distribution around mean, not really "error of mean"
    ### plotting
    plt.figure(figsize=(9.3,6))
    plt.bar(bar_centers, data_boots, width=bar_size, color=['cornflowerblue','royalblue'], label="bootstrap bars")    # histogram from bootstrap
    plt.errorbar(bar_centers, data_boots, fmt=" ", yerr=error_boots, color='black', capsize=2, label="bar error")      # error-bars from bootstrap
    plt.axvline(x=calc_mean_mean[0], color='red', label="original mean,\nwith error as red bar")       # visaulizing the bootstrap calculated mean
    plt.bar(calc_mean_mean[0], 40, width=calc_mean_mean[1],facecolor='r',alpha=0.2)     # error of mean
    plt.axvline(x=mean_boot_samples, linestyle='dashed', color='limegreen', label="bootstrap mean,\nwith error as green bar")       # visaulizing the beforehand calculated mean
    plt.bar(mean_boot_samples, 40, width=err_mean_boot,facecolor='g',alpha=0.1)     # error of mean
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="upper left")
    plt.ylim(0,35)
    plt.xlabel('value')
    plt.ylabel('counts')
    if TYPE == 0:
        plt.xlim(7020,7320)
        plt.annotate('original mean: '+ str(calc_mean_mean[0])+' +- '+str(calc_mean_mean[1]),(90,270),xycoords='figure points')
        plt.annotate('bootstrap mean: '+ str(np.round(mean_boot_samples,3))+' +- '+str(np.round(err_mean_boot,3)),(90,250),xycoords='figure points')
        plt.title('Magnetization histogram via bootstrap-method')
    if TYPE == 1:
        plt.xlim(-6290,-6180)
        #plt.xlim(-6395,-6075) # same delta x as magn plot
        plt.annotate('original mean: '+ str(calc_mean_mean[0])+' +- '+str(calc_mean_mean[1]),(90,270),xycoords='figure points')
        plt.annotate('bootstrap mean: '+ str(np.round(mean_boot_samples,3))+' +- '+str(np.round(err_mean_boot,3)),(90,250),xycoords='figure points')
        plt.title('Energy histogram via bootstrap-method')
        #plt.title('Energy histogram via bootstrap-method (with the same $\Delta x$ as magnetization plot)')
    plt.show()




''' --- run the code --- '''
### initiate which array
data_magn = np.loadtxt('Results/mean_mag.csv') # magn mean data
data_energy = np.loadtxt('Results/mean_energies.csv') # energies mean data
### define what is the beforehand calculated mean
calc_mean_mean_magn = np.array([7181.257,2.111])
calc_mean_mean_energy = np.array([-6235.977,0.629])
### set number of bars for histograms
num_bars = 50
### set parameters for bootstrap-method
numb_samples = 100
size_small_sample = 5
print('Parameters used for bootstrap-method: random.seed={3}, #bars={0}, #boot samples={1}, size small samples={2}'.format(num_bars,numb_samples,size_small_sample,boot_seed))


# magnetization
final_histogramm(0,data_magn,calc_mean_mean_magn,num_bars,numb_samples,size_small_sample)
# energy
final_histogramm(1,data_energy,calc_mean_mean_energy,num_bars,numb_samples,size_small_sample)






''' --- here are the parts used which are in the appendix of the report --- '''

def orig_histo_and_Gauss(data):
    #### for checking: histogram using original replicas without bootstrap
    y,binEdges = np.histogram(data,bins=num_bars)
    bar_centers = 0.5*(binEdges[1:]+binEdges[:-1])
    bar_size = ((np.max(binEdges)-np.min(binEdges))/num_bars)
    plt.bar(bar_centers,y, width=bar_size, color='green')     
    #### distributiuon using gaussian kernel
    kde = gaussian_kde(data)
    dist_space = linspace(min(data),max(data),100)
    plt.plot(dist_space,kde(dist_space)/np.max(kde(dist_space))*(np.max(y)), color='purple')
    #### plot
    plt.title('histogram of original 500 replicas and gaussian "fit"')
    plt.xlabel('value')
    plt.ylabel('counts')
    #plt.show() # delete the # if you want to plot this 

orig_histo_and_Gauss(data_magn)
orig_histo_and_Gauss(data_energy)


