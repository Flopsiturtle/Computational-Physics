import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Nth = 200
R = 500


'lose all values below thermalization'

column_names = ['replica_number' + str(i+1) for i in range(R)]
energydata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\EnergyReplica.csv', names = column_names, index_col=False) #change path accordingly
energydata.index = energydata.index + 1
x = energydata.index.values
indexnames = energydata[energydata.index < Nth].index
energydata.drop(indexnames, inplace =True)


'calculate mean energies'

mean_energies = []
for i in range(R):
    values = energydata['replica_number' + str(i+1)]
    mean = np.sum(values)/len(values)
    mean_energies.append(mean)
    
'calculate expectation values and statistical error'

exp_value = 1/len(mean_energies) * np.sum(mean_energies)
deviation = (mean_energies - exp_value)**2
stat_err = np.sqrt(np.sum(deviation) / (R*(R - 1)))
print(exp_value)
print(stat_err)




Nth = 200
R = 500


'lose all values below thermalization'

magdata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\MagnetizationReplica.csv', names = column_names, index_col=False)
magdata.index = magdata.index + 1
x = magdata.index.values
indexnames = magdata[magdata.index < Nth].index
magdata.drop(indexnames, inplace =True)


'calculate mean energies'

mean_mag = []
for i in range(R):
    values = magdata['replica_number' + str(i+1)]
    mean = np.sum(values)/len(values)
    mean_mag.append(mean)
    
'calculate expectation values and statistical error'

exp_value = 1/len(mean_mag) * np.sum(mean_mag)
deviation = (mean_mag - exp_value)**2
stat_err = np.sqrt(np.sum(deviation) / (R*(R - 1)))
print(exp_value)
print(stat_err)






