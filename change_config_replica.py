import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Nth = 200
R = 500
u = 100 #max number of configurations

'lose all values below thermalization'

column_names = ['replica_number' + str(i+1) for i in range(R)]
energydata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\EnergyReplica.csv', names = column_names, index_col=False) #change path accordingly
energydata.index = energydata.index + 1
x = energydata.index.values
indexnames = energydata[energydata.index < Nth].index
energydata.drop(indexnames, inplace =True)
stat_error = []
for i in range(u):
    w = energydata.copy()
    indexnames = energydata[energydata.index > Nth + i].index
    w.drop(indexnames, inplace =True)
    
    'calculate mean energies'
    
    mean_energies = []
    for i in range(R):
        values = w['replica_number' + str(i+1)]
        mean = np.sum(values)/len(values)
        mean_energies.append(mean)
        
    'calculate expectation values and statistical error'
    
    exp_value = 1/len(mean_energies) * np.sum(mean_energies)
    deviation = (mean_energies - exp_value)**2
    stat_err = np.sqrt(np.sum(deviation) / (R*(R - 1)))
    stat_error.append(stat_err)
    
'plotting'

x = [i +1 for i in range(u)]
plt.plot(x, stat_error)

