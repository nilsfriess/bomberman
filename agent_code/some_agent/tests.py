import numpy as np

import more_helpers

l = [(2,3,7),(4,5,7)]
c = [(x,y) for (x,y,_) in l]

a = np.array(((2,3),(3,5)))

try:
    if a[4,1] == 2:
        print("fine")
except IndexError:
    print("also fine")

t = np.zeros((7,7))
for (_,(x,y)) in more_helpers.get_step_neighbourhood(0,0,2):
    t[x+3,y+3] = 1
print(t)
