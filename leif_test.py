class hello:
    def __init__(self, a):
        self.k = a
    def fun(self):
        return self.k

a =  [[hello(2), hello(1)]]
print(a[0][0].fun())

b = [[]*3]*2

print(b)
import numpy as np

c = np.array([1,2,3])
print(c[1:])

x = [c,c]
y = np.array(x)
print(y)
