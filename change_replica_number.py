import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Nth = 200
R = 100

'Energy'
'lose all values below thermalization'

column_names = ['replica_number' + str(i+1) for i in range(R)]
energydata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\EnergyReplica.csv', names = column_names, index_col=False) #change path accordingly
energydata.index = energydata.index + 1
x = energydata.index.values
indexnames = energydata[energydata.index < Nth].index
energydata.drop(indexnames, inplace =True)

stat_error = []

for j in [i + 2 for i in range(R-2)]:
    
    'calculate mean energies'
    
    mean_energies = []
    for i in range(j):
        values = energydata['replica_number' + str(i+1)]
        mean = np.sum(values)/len(values)
        mean_energies.append(mean)
        
    'calculate expectation values and statistical error'
    
    exp_value = 1/len(mean_energies) * np.sum(mean_energies)
    deviation = (mean_energies - exp_value)**2
    stat_err = np.sqrt(np.sum(deviation) / (j*(j - 1)))
    stat_error.append(stat_err)
    
x = [i + 2 for i in range(R-2)]
plt.plot(x, stat_error)
plt.show()


'magnetization'
'lose all values below thermalization'

column_names = ['replica_number' + str(i+1) for i in range(R)]
energydata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\MagnetizationReplica.csv', names = column_names, index_col=False) #change path accordingly
energydata.index = energydata.index + 1
x = energydata.index.values
indexnames = energydata[energydata.index < Nth].index
energydata.drop(indexnames, inplace =True)

stat_error = []

for j in [i + 2 for i in range(R-2)]:
    
    'calculate mean energies'
    
    mean_energies = []
    for i in range(j):
        values = energydata['replica_number' + str(i+1)]
        mean = np.sum(values)/len(values)
        mean_energies.append(mean)
        
    'calculate expectation values and statistical error'
    
    exp_value = 1/len(mean_energies) * np.sum(mean_energies)
    deviation = (mean_energies - exp_value)**2
    stat_err = np.sqrt(np.sum(deviation) / (j*(j - 1)))
    stat_error.append(stat_err)
    
x = [i + 2 for i in range(R-2)]
plt.plot(x, stat_error)