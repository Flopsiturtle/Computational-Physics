import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15, 10))
ax1.set_title(r'$B=0,005 \: \beta = 0,454545$')
ax2.set_title(r'$B=0,005 \: \beta = 0,5$')
ax3.set_title(r'$B=0,01  \: \beta = 1$')
ax4.set_title(r'$B=0,01 \: \beta = 1,666$')
for i in [ax1, ax2, ax3, ax4]:
    i.set_xlabel('configuration number')
    i.set_ylabel('magnetisation')


R = 500


column_names = ['replica_number' + str(i+1) for i in range(R+1)]
energydata =  pd.read_csv('C:\\Users\\Mickey\\Desktop\\Computational-Physics-1\\Results3\\Results3\\MagB0.000500Beta0.454545.csv', names = column_names, index_col=False) #change path accordingly
energydata = energydata.drop(['replica_number' + str(R+1)], axis = 1)
energydata.index = energydata.index + 1
x = energydata.index.values
#indexnames = energydata[energydata.index < Nth].index
#energydata.drop(indexnames, inplace =True)
for i in column_names[:-1]:
    ax1.plot(x, energydata[i])



column_names = ['replica_number' + str(i+1) for i in range(R+1)]
energydata =  pd.read_csv('C:\\Users\\Mickey\\Desktop\\Computational-Physics-1\\Results3\\Results3\\MagB0.000500Beta0.500000.csv', names = column_names, index_col=False) #change path accordingly
energydata = energydata.drop(['replica_number' + str(R+1)], axis = 1)
energydata.index = energydata.index + 1
x = energydata.index.values
#indexnames = energydata[energydata.index < Nth].index
#energydata.drop(indexnames, inplace =True)
for i in column_names[:-1]:
    ax2.plot(x, energydata[i])

column_names = ['replica_number' + str(i+1) for i in range(R+1)]
energydata =  pd.read_csv('C:\\Users\\Mickey\\Desktop\\Computational-Physics-1\\Results3\\Results3\\MagB0.010000Beta1.000000.csv', names = column_names, index_col=False) #change path accordingly
energydata = energydata.drop(['replica_number' + str(R+1)], axis = 1)
energydata.index = energydata.index + 1
x = energydata.index.values
#indexnames = energydata[energydata.index < Nth].index
#energydata.drop(indexnames, inplace =True)
for i in column_names[:-1]:
    ax3.plot(x, energydata[i])
    
column_names = ['replica_number' + str(i+1) for i in range(R+1)]
energydata =  pd.read_csv('C:\\Users\\Mickey\\Desktop\\Computational-Physics-1\\Results3\\Results3\\MagB0.010000Beta1.666667.csv', names = column_names, index_col=False) #change path accordingly
energydata = energydata.drop(['replica_number' + str(R+1)], axis = 1)
energydata.index = energydata.index + 1
x = energydata.index.values
#indexnames = energydata[energydata.index < Nth].index
#energydata.drop(indexnames, inplace =True)
for i in column_names[:-1]:
    ax4.plot(x, energydata[i])