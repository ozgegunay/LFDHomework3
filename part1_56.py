#Özge Günay 150114027

import math
import sympy as s
import numpy as np

#Initial weights
w = [1, 1]

u = s.Symbol('u')
v = s.Symbol('v')
errorFunction = (u*s.exp(v) - 2*v*s.exp(-u))**2

#Calculating the direction
def calculateDirection(w):
    deru = s.diff(errorFunction, u)
    derv = s.diff(errorFunction, v)
    gradu = deru.evalf(subs={u: w[0], v: w[1]})
    gradv = derv.evalf(subs={u: w[0], v: w[1]})
    grad = [gradu, gradv]
    vt = [-1 * x for x in grad]
    return vt

iteration=0
learn_rate = 0.1

#Learning process
while (errorFunction.evalf(subs={u:w[0], v:w[1]})>=10**-14):
    iteration += 1
    direction = calculateDirection(w)
    newDirec = [learn_rate * x for x in direction]
    w = np.add(w, newDirec)

print("The final weight is " + str(w))
print("The number of iteration is " + str(iteration))