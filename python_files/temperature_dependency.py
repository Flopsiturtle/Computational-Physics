import glob 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Nth = 200
R = 500
fig, ((ax01, ax02), (ax1, ax2), (ax3, ax4)) = plt.subplots(3,2,figsize=(15,11))
ax01.set_title('energy-density')
ax02.set_title('magnetisation-density')

for i in [ax01, ax02, ax1, ax2, ax3, ax4]:
    i.set_xlabel(r'$1/ \beta$', fontsize = 10)
    i.tick_params(axis='both', which='major', labelsize=10)
ax01.set_ylabel('$<E>/V$', fontsize = 10)    
ax02.set_ylabel('$<M>/V$', fontsize = 10)    
ax1.set_ylabel('$<E>/V$', fontsize = 10)    
ax2.set_ylabel('$<M>/V$', fontsize = 10)    
ax3.set_ylabel('$<E>/V$', fontsize = 10)    
ax4.set_ylabel('$<M>/V$', fontsize = 10)  

'energy'

for B in ['0.000500' , '0.001000', '0.005000',  '0.010000']:
    exp_value_list = []
    stat_err_list = []
    beta_list = []
    names = []
    for name in glob.glob('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\Results2\\EnB' + str(B) +'Beta*.csv'):
        names.append(name)
        beta_list.append(float(name[-12:-4]))   #watch out this needs to work with the given beta values
        column_names = ['replica_number' + str(i+1) for i in range(R+1)]
        energydata =  pd.read_csv(name, names = column_names, index_col=False) 
        energydata = energydata.drop(["replica_number" + str(R+1)],axis = 1)
        energydata.index = energydata.index + 1
        x = energydata.index.values
        indexnames = energydata[energydata.index < Nth].index
        energydata.drop(indexnames, inplace =True)
        mean_energies = []
        for i in range(R):
            values = energydata['replica_number' + str(i+1)]
            mean = np.sum(values)/len(values)
            mean_energies.append(mean)
        exp_value = 1/len(mean_energies) * np.sum(mean_energies)
        deviation = (mean_energies - exp_value)**2
        stat_err = np.sqrt(np.sum(deviation) / (R*(R - 1)))
        exp_value_list.append(exp_value)
        stat_err_list.append(stat_err)
    
    exp_value_list = np.array(exp_value_list)
    stat_err_list = np.array(stat_err_list)
    ax1.errorbar(1/np.array(beta_list), exp_value_list/10000, yerr = stat_err_list/10000, fmt = 'o', label = '$B=$'+ str(float(B)))
    ax3.set_xlim(1.5,4)
    ax3.set_ylim(-1, -0.1)
    ax3.errorbar(1/np.array(beta_list), exp_value_list/10000, yerr = stat_err_list/10000, fmt = 'o', label = '$B=$'+ str(float(B)))
    if B == '0.010000':
        ax01.errorbar(1/np.array(beta_list), exp_value_list/10000, yerr = stat_err_list/10000, fmt = 'o', label = '$B=$'+ str(float(B)))

    

'magnetisation'

for B in ['0.000500' , '0.001000', '0.005000',  '0.010000']:
    exp_value_list = []
    stat_err_list = []
    beta_list = []
    names = []
    for name in glob.glob('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\Results2\\MagB' + str(B) +'Beta*.csv'):
        names.append(name)
        beta_list.append(float(name[-12:-4]))   #watch out this needs to work with the given beta values
        column_names = ['replica_number' + str(i+1) for i in range(R+1)]
        energydata =  pd.read_csv(name, names = column_names, index_col=False) 
        energydata = energydata.drop(["replica_number" + str(R+1)],axis = 1)
        energydata.index = energydata.index + 1
        x = energydata.index.values
        indexnames = energydata[energydata.index < Nth].index
        energydata.drop(indexnames, inplace =True)
        mean_energies = []
        for i in range(R):
            values = energydata['replica_number' + str(i+1)]
            mean = np.sum(values)/len(values)
            mean_energies.append(mean)
        exp_value = 1/len(mean_energies) * np.sum(mean_energies)
        deviation = (mean_energies - exp_value)**2
        stat_err = np.sqrt(np.sum(deviation) / (R*(R - 1)))
        exp_value_list.append(exp_value)
        stat_err_list.append(stat_err)
    
    exp_value_list = np.array(exp_value_list)
    stat_err_list = np.array(stat_err_list)
    ax2.errorbar(1/np.array(beta_list), exp_value_list/10000, yerr = stat_err_list/10000, fmt = 'o', label = '$B=$'+ str(float(B)))
    ax4.set_xlim(0.1,4 )
    ax4.set_ylim(0,1)
    ax4.errorbar(1/np.array(beta_list), exp_value_list/10000, yerr = stat_err_list/10000, fmt = 'o', ls = '-', label = '$B=$'+ str(float(B)))
    if B == '0.010000':
        ax02.errorbar(1/np.array(beta_list), exp_value_list/10000, yerr = stat_err_list/10000, fmt = 'o', label = '$B=$'+ str(float(B)))
        
for i in [ax01, ax02, ax1, ax2, ax3, ax4]:
    i.legend()
        
        
        
        
        
        
        
        