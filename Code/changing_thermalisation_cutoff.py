import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

R = 500

'plot development depending on Nth'

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('varying the thermalization cut-off', fontsize = 20)
for i in [ax1, ax2, ax3, ax4]:
    i.set_xlabel('cut-off configuration', fontsize = 10)
    i.tick_params(axis='both', which='major', labelsize=10)
ax1.set_ylabel('energy mean', fontsize = 10)    
ax2.set_ylabel('magnetisation mean', fontsize = 10)    
ax3.set_ylabel('energy mean statistical error', fontsize = 10)    
ax4.set_ylabel('magnetisation mean statistical error', fontsize = 10)    



E = []
E_err = []
M = []
M_err = []

N_values = np.array([2*i+2 for i in range(480)])

for Nth in N_values:
    
    column_names = ['replica_number' + str(i+1) for i in range(R+1)]
    energydata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\EnergyReplica.csv', names = column_names, index_col=False) #change path accordingly
    energydata = energydata.drop(['replica_number' + str(R+1)], axis = 1)
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
    E.append(exp_value)
    E_err.append(stat_err)
    
    magdata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\MagnetizationReplica.csv', names = column_names, index_col=False)
    magdata = magdata.drop(['replica_number' + str(R+1)], axis = 1)
    magdata.index = magdata.index + 1
    x = magdata.index.values
    indexnames = magdata[magdata.index < Nth].index
    magdata.drop(indexnames, inplace =True)

    mean_mag = []
    for i in range(R):
        values = magdata['replica_number' + str(i+1)]
        mean = np.sum(values)/len(values)
        mean_mag.append(mean)
        

    exp_value = 1/len(mean_mag) * np.sum(mean_mag)
    deviation = (mean_mag - exp_value)**2
    stat_err = np.sqrt(np.sum(deviation) / (R*(R - 1)))
    M.append(exp_value)
    M_err.append(stat_err)


ax1.errorbar(N_values, E, E_err, fmt="none")
ax2.errorbar(N_values, M, M_err, fmt="none", color = 'r')
ax3.scatter(N_values, E_err, s=5)
ax4.scatter(N_values, M_err, s=5, color = 'r')

