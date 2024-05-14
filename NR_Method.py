# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:09:27 2024

@author: Suresh BASNET
"""

import numpy as np
import matplotlib.pyplot as plt

#To find the root of the equation: f(x) = x^3+2x^2+5x+1

x0 = 5  # Initial guess
tol = 1e-8   # Tolerance for convergence
nx = 1000  # Maximum number of iterations

#=====Define function
def f(x):
    return x**3 +2*x**2+5*x +1 

#====Define derivative of function
def df(x):
    return 3*x**2+4*x+5

#====Function definition in Newton's-Raphson method
def NR(x0, tol, nx):
    x_old = x0
    
    for i in range(nx):
        x_new = x_old - f(x_old) / df(x_old)
        if abs(x_new - x_old) < tol:
            print(f"Root found at x = {x_new} after {i+1} iterations.")
            return x_new
        elif abs(f(x_new)) < tol:
            print(f"Root found at x = {x_new} after {i+1} iterations (functional tolerance).")
            return x_new
        elif abs(df(x_new)) < tol:
            print(f"Root found at x = {x_new} after {i+1} iterations (derivative tolerance).")
            return x_new
        x_old = x_new
    print("Root not found within the maximum number of iterations.")
    return None

root = NR(x0, tol, nx)
#root = round(root,4)
x_values = np.linspace(-10, 10, nx)
# Plotting f(x) vs x
plt.plot(x_values, f(x_values), label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axvline(root, color='r', linestyle='--', label=f'Root at x={root}')
plt.scatter(root, f(root), color='r')
plt.legend()
plt.grid(True)
plt.show()
