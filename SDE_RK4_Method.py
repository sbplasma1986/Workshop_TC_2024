# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:56:27 2024

@author: Suresh BASNET
"""

import numpy as np
import matplotlib.pyplot as plt

#2nd order Differential equation using R-K 4th order method
# d^2y/dx^2+ydy/dx+3y-sin(x) = 0


# Initial and final values of x
x_initial = 0
x_final = 10

# Number of steps
nx = 20

# Calculate step size
h = (x_final - x_initial) / nx


# Define the function representing the system of first-order ODEs
def func(x, y):
    dydx = y[1]  # dy/dx = v
    dvdx = np.sin(x) - y[0] * y[1] - 3 * y[0] # dv/dx = sin(x) - y*v - 3*y
    return np.array([dydx, dvdx])

# Fourth-order Runge-Kutta method implementation
def RK4(f, x0, y0, h, n):
 # Initializes the position and functional values
    x_values = [x0]
    y_values = [y0]
    
    for i in range(nx):
        xn = x_values[-1]
        yn = y_values[-1]
        
        k1 = h * f(xn, yn)
        k2 = h * f(xn + 0.5*h, yn + 0.5*k1)
        k3 = h * f(xn + 0.5*h, yn + 0.5*k2)
        k4 = h * f(xn + h, yn + k3)
        
        y_new = yn + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_new = xn + h
        
        x_values.append(x_new)
        y_values.append(y_new)
    
    return x_values, y_values


# Initial conditions
y0 = np.array([-1, 1])  # y(x = 0) = -1, dy/dx(x=0) = 1

# Run the fourth-order Runge-Kutta method
x_values, y_values = RK4(func, x_initial, y0, h, nx)

# Extract y and dy/dx values
y_v = np.array(y_values)[:, 0]
y_p = np.array(y_values)[:, 1]


# Plot the result
plt.plot(x_values, y_v, label='Approximate Solution')
plt.plot(x_values, y_p, label='dy/dx')
plt.xlabel('x-values')
plt.ylabel('y(x) and dy/dx')
plt.title('Solution of the Second-Order ODE using Runge-Kutta 4th Order')
plt.grid(True)
plt.legend()
plt.show()
