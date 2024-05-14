# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:40:25 2024

@author: Suresh BASNET
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
a = 0.05
b = 1
c = 0.1
d = 0.5

# Define the coupled differential equations
def func(x, y):
    R, y_val = y
    dR_dx = a * R - b * R * y_val
    dy_dx = -c * y_val + d * R * y_val
    return np.array([dR_dx, dy_dx])

# Fourth-order Runge-Kutta method implementation
def runge_kutta4(f, x0, y0, h, n):
    """
    f: function representing the coupled differential equations
    x0: initial value of x
    y0: initial value of y(x0)
    h: step size
    n: number of steps
    """
    x_values = [x0]
    y_values = [y0]
    
    for i in range(n):
        x = x_values[-1]
        y = y_values[-1]
        
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5*h, y + 0.5*k1)
        k3 = h * f(x + 0.5*h, y + 0.5*k2)
        k4 = h * f(x + h, y + k3)
        
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_next = x + h
        
        x_values.append(x_next)
        y_values.append(y_next)
    
    return x_values, y_values

# Initial conditions
x0 = 2
R0 = 15
y0 = 20

# Step size and number of steps
h = 0.01
n = 10000

# Run the fourth-order Runge-Kutta method
x_values, y_values = runge_kutta4(func, x0, [R0, y0], h, n)

# Extract R values and y values
R_values = [y[0] for y in y_values]
y_values = [y[1] for y in y_values]

# Plot the results
plt.plot(x_values, R_values, label='R(x)')
plt.plot(x_values, y_values, label='y(x)')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Solution of Coupled Differential Equations')
plt.legend()
plt.grid(True)
plt.show()
