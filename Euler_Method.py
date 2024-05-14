# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:06:07 2024

@author: Suresh BASNET
"""

import numpy as np
import matplotlib.pyplot as plt


# Define time interval, initial conditions and step size
nt = 100
t0 = 0
tf = 10
h = (tf-t0)/nt
f0 = -1


# Define the function representing the derivative df/dt = exp(-t)
def func(t):
    return np.exp(-t)


# Define the Euler method function to solve the differential equation
def EM(func, t0, f0, h, nt):
 # Initializes the time and functional values
    t_values = [t0]
    f_values = [f0]
    for i in range(nt):
        t_new = t_values[-1] + h
        f_new = f_values[-1] + h * func(t_values[-1])
        t_values.append(t_new)
        f_values.append(f_new)
    return t_values, f_values

# Solve the differential equation using the Euler method
t_values, f_values = EM(func, t0, f0, h, nt)

# Plot the solution
plt.plot(t_values, f_values, label='Euler Method')
plt.xlabel('Time, t')
plt.ylabel('Functional value, f(t)')
#plt.title('Euler Method Solution for df/dt = exp(-t)')
plt.legend()
plt.grid(True)
plt.show()
