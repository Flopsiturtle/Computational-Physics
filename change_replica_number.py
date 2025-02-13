import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

Nth = 200
R = 500


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(12, 8))
fig.suptitle('varying replica number', fontsize = 20)
for i in [ax1, ax2, ax3, ax4]:
    i.set_xlabel('replica number', fontsize = 10)
    i.set_ylabel('statistical error', fontsize = 10)
    i.tick_params(axis='both', which='major', labelsize=10)
    
def f(x, m, n):
    return m*x + n
   

'Energy'


column_names = ['replica_number' + str(i+1) for i in range(R+1)]
energydata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\EnergyReplica.csv', names = column_names, index_col=False) #change path accordingly
energydata = energydata.drop(['replica_number' + str(R+1)], axis = 1)
energydata.index = energydata.index + 1
x = energydata.index.values
indexnames = energydata[energydata.index < Nth].index
energydata.drop(indexnames, inplace =True)


stat_error = []
linearised = []

for j in [i + 2 for i in range(R-2)]:
    
    
    mean_energies = []
    for i in range(j):
        values = energydata['replica_number' + str(i+1)]
        mean = np.sum(values)/len(values)
        mean_energies.append(mean)
        

    
    exp_value = 1/len(mean_energies) * np.sum(mean_energies)
    deviation = (mean_energies - exp_value)**2
    stat_err = np.sqrt(np.sum(deviation) / (j*(j - 1)))
    stat_error.append(stat_err)
    linearised.append(1/(stat_err)**2)

    
x = np.array([i + 2 for i in range(R-2)])
popt, pcov = curve_fit(f, x, linearised)
y = f(x, *popt)
ax1.plot(x, stat_error,  label = 'Energy error')
ax3.plot(x, linearised, label = 'linearised energy error \n of the form $1/y^2 $')
ax3.plot(x, y, label =  'linear fit $f=mx+n$')
print('Energy:')
print('m = ' + str(popt[0]) + ' +/- ' + str(np.sqrt(pcov[0][0])))
print('n = ' + str(popt[1]) + ' +/- ' + str(np.sqrt(pcov[1][1])))


'magnetization'

column_names = ['replica_number' + str(i+1) for i in range(R+1)]
energydata =  pd.read_csv('C:\\Users\\Mickey Wilke\\Desktop\\cp2_ising\\MagnetizationReplica.csv', names = column_names, index_col=False) #change path accordingly
energydata = energydata.drop(['replica_number' + str(R+1)], axis = 1)
energydata.index = energydata.index + 1
x = energydata.index.values
indexnames = energydata[energydata.index < Nth].index
energydata.drop(indexnames, inplace =True)

stat_error = []
linearised = []

for j in [i + 2 for i in range(R-2)]:
    
    
    mean_energies = []
    for i in range(j):
        values = energydata['replica_number' + str(i+1)]
        mean = np.sum(values)/len(values)
        mean_energies.append(mean)
        
    
    exp_value = 1/len(mean_energies) * np.sum(mean_energies)
    deviation = (mean_energies - exp_value)**2
    stat_err = np.sqrt(np.sum(deviation) / (j*(j - 1)))
    stat_error.append(stat_err)
    linearised.append(1/(stat_err)**2)

    
x = np.array([i + 2 for i in range(R-2)])
popt, pcov = curve_fit(f, x, linearised)
y = f(x, *popt)
ax2.plot(x, stat_error, label = 'Magnetisation error', color = 'r')
ax4.plot(x, linearised, label = 'linearised energy error \n of the form $1/y^2 $', color = 'r')
ax4.plot(x, y, label =  'linear fit $f=mx+n$', color = 'g')
print('Magnetisation:')
print('m = ' + str(popt[0]) + ' +/- ' + str(np.sqrt(pcov[0][0])))
print('n = ' + str(popt[1]) + ' +/- ' + str(np.sqrt(pcov[1][1])))


ax1.legend(fontsize = 10)
ax2.legend(fontsize = 10)
ax3.legend(fontsize = 10, loc = 'lower right')
ax4.legend(fontsize = 10, loc = 'lower right')
