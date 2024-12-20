import numpy as np
import matplotlib.pyplot as plt

import eigenmethods


#print(eigenmethods.arnoldi(np.ones(190), 4, 0.000001, 500, 0.000001, 500)[0])


#### !!!!!!!for some N we get an error!!!!!!!


#def get_eigen_array(number_eigen,start,finish,gap):
#    store_eigen = []
#    for N in np.arange(start,finish+gap,gap):
#        v = np.ones(N)
#        #print(eigenmethods.arnoldi(v, 1, 0.0001, 100, 0.0001, 200)[0])
#        data = eigenmethods.arnoldi(v, number_eigen, 0.0001, 500, 0.0001, 500)[0]
#        if isinstance(data,str) is False: 
#            store_eigen.append((data,N))
#    return store_eigen
#a = get_eigen_array(1,50,100,10)
#x=[]
#y=[]
#for i in np.arange(len(a)):
#    x.append(a[i][1])
#    y.append(a[i][0])



def get_data(number_eigen,start,finish,gap,mu,epsilon):
    x = []
    y = []
    for N in np.arange(start,finish+gap,gap):
        v = np.ones(N)
        #print(eigenmethods.arnoldi(v, 1, 0.0001, 100, 0.0001, 200)[0])
        data = eigenmethods.arnoldi(v, number_eigen, 0.000001, 500, 0.000001, 500,mu,epsilon)[0]
        if isinstance(data,str) is False: 
            x.append(N)
            y.append(data)
    return x,y


mu = 153.9     # our mu
#mu = 20         # his mu
epsilon = 1/60

data = get_data(4,200,240,40,mu,epsilon)     ### maybe delete input with start,finish,gap to be always the same and give input of errors and maxiters, depending on usecase we want for final test
plt.plot(data[0],data[1],'ro')
plt.show()

### i think if eigenvalues weird just make error smaller maybe


