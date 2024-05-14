# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:22:19 2024

@author: Suresh BASNET
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
a = 10
b = 1.5

# Initial conditions
t0 = 0
f0 = 2
#f0 = float(input('f0 = '))
#======Step size 
tf = 10
n = 50
h = (tf-t0)/n


# Define the differential equation df/dt = a*t^2 + b
def func(t, f):
    return a * t**2 + b

# Fourth-order Runge-Kutta method implementation
def RK4(f, t0, f0, h, n):
    
# Initializes the time and functional values
    t_values = [t0]
    f_values = [f0]
    
    for i in range(n):
        tn = t_values[-1]
        fn = f_values[-1]  # Renamed the variable to avoid conflicts
        
        k1 = h * f(tn, fn)  # Fixed the function call
        k2 = h * f(tn + 0.5*h, fn + 0.5*k1)
        k3 = h * f(tn + 0.5*h, fn + 0.5*k2)
        k4 = h * f(tn + h, fn + k3)
        
        f_new = fn + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_new = tn + h
        
        t_values.append(t_new)
        f_values.append(f_new)
    
    return t_values, f_values


# Run the fourth-order Runge-Kutta method
t_values, f_values = RK4(func, t0, f0, h, n)

# Plot the results
plt.rcParams.update({'font.family':'Times New Roman'})
plt.plot(t_values, f_values, label='Runge-Kutta 4th Order')
plt.legend()
plt.legend(loc='upper left',prop = {'size': 14})
plt.xlim(0,10)
plt.ylim(0,3500)
plt.xlabel(r'Time values ( t )', fontname='Times New Roman',
           fontsize=16,color='red')
plt.ylabel('Functional value f(t)', fontsize=16,color='black')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#plt.title('Straight line', fontsize=14, color='blue')
#=============Add minor ticks on the plot
plt.minorticks_on()
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
#========================================
#plt.savefig('RK4.png',dpi=600,bbox_inches='tight')
plt.grid(True)
plt.show()

    