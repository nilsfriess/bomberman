import numpy as np

l = [1,2,3,4,5]
print(l[0:len(l)-2])
print(l[len(l)-2:])

a = np.array(l)
b = np.array([a])
c = a.reshape(1,-1)
print(c)
print(b)
