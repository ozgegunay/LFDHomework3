#Özge Günay 150114027

import math
import sympy as s
import numpy as np

#Initial weights
w = [1, 1]

u = s.Symbol('u')
v = s.Symbol('v')
errorFunction = (u*s.exp(v) - 2*v*s.exp(-u))**2

#Derivations of error function with respect to u and v
deru = s.diff(errorFunction, u)
derv = s.diff(errorFunction, v)

#Learning rate
learn_rate = 0.1
error = errorFunction.evalf(subs={u:w[0], v:w[1]})

for x in range(15):

    w[0] = w[0] - learn_rate * deru.evalf(subs={u: w[0], v: w[1]})
    w[1] = w[1] - learn_rate * derv.evalf(subs={u: w[0], v: w[1]})
    
    #print(errorFunction.evalf(subs={u:w[0], v:w[1]}))

print("The error after the 15 iteration is " + str(errorFunction.evalf(subs={u:w[0], v:w[1]})))
