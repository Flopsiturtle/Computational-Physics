import glob 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Nth = 200
R = 500

exp_value_list = []
stat_err_list = []
beta_list = np.linspace(0.2,10,24)
for name in glob.glob('C:\\Users\\Mickey\\Desktop\\Computational-Physics\\Results\\EnB0.010000Beta*.csv'):
    column_names = ['replica_number' + str(i+1) for i in range(R)]
    energydata =  pd.read_csv(name, names = column_names, index_col=False) #change path accordingly
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
    
exp_value_list = np.flip(np.array(exp_value_list))
stat_err_list = np.flip(np.array(stat_err_list))
plt.errorbar(beta_list, exp_value_list/10000, yerr = stat_err_list/10000, fmt = 'o')