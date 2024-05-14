# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:16:41 2024

@author: Suresh BASNET
"""
import sympy as sym
import numpy as np
import math as math
from scipy.integrate import quad  # Use this to find the finite integral 

# Define a symbolic expression

x = sym.Symbol('x')
y = sym.Symbol('y')

Fxy = x + 2*y

#Fxy1 = sym.simplify((x + x * y) / x)


Fxy2 = sym.limit(sym.sin(x) / x, x, 0)    # Calculus
 
#Fxy3 = sym.expand((x + y) ** 3)

#Fxy4 = sym.expand(sym.cos(x + y), trig=True)  # Compound Angle Formula


#======================Differentiation

Fx = sym.sin(x)+x**2 + 1

Fx1 = y*x**2 + 1 + sym.sin(x)

dFx = sym.diff(Fx, x)

dFx1 = sym.diff(Fx1,x)

#===========Perform higher order derivatives
Fx2 = sym.diff(Fx1, x, 1)

#========================Indefinite and finite Integration
Fx4 = sym.integrate(Fx, x)

Fxx = sym.cos(x)

Fx5 = sym.integrate(Fx, (x, -1, 1))

#=================Finite function

IFx = sym.integrate(Fxx, x)

# Calculate the IFx at x = -1
LL = sym.limit(IFx, x, -1)

# Calculate the IFx at x = 1
UL = sym.limit(IFx, x, 1)





