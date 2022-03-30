#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:32:47 2021

@author: gabryelsr
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-297.5192,297.5192)
pl=4.7/100
#T=pl*9.8*max(x)
#a=pl/T
y=[]

d=8e-3
A=(np.pi*d**4)/4
rho=4.7/(100*A)
E=11.6e8
L=301.6914
T=pl*L
w=pl-A*1.225
h=41

zm=(L/8)*(w*L/T)*(1+(T/(E*A)))
print(zm)

Lmax=np.sqrt(8*T*h/(w*(1+(T/(E*A)))))
print(Lmax)

T0=w*(L**2)/(8*h-(w*(L**2))/E*A)
print(T0)



for pos in x:
    T=pl*9.8*pos
    a=pl/T
    alt=(1/a)*np.cosh(a*pos)
    y.append(alt)
    
plt.plot(x,y)
plt.grid()
plt.show()