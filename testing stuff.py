import numpy as np

aa = np.array([[0,1,2],[5,6,7],[11,12,13]])
print(aa)
shape_a = aa.shape

print((aa.shape)[0])

print(np.zeros(len(shape_a)))

print(aa[1,1])


bb = np.array([[[[0,1,2],[5,6,7],[11,12,13]]],[[[0,1,2],[5,6,7],[11,12,13]]],[[[0,1,2],[5,6,7],[11,12,13]]]])

print(np.multiply(aa,bb))

print(np.arange(2))

D = len(aa.shape)
n = np.zeros(D)
n_vector = np.zeros(aa.shape)

#for i in np.arange(D):
#    for j in np.arange((aa.shape)[i]):
#        n_vector[j] = 
#print(n_vector)

print(np.mgrid[:5,:5].transpose(1,2,0))

print(bb@bb)