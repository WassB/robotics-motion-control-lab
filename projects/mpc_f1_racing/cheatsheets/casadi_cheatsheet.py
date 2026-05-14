"""
CasADi Cheat Sheet - made for EL2700

A concatenation of the following:
http://casadi.sourceforge.net/cheatsheets/python.pdf
https://github.com/casadi/casadi/wiki/cheatsheets

All rights reserved to CasADi's authors.

CasADi Documentation:
https://web.casadi.org/docs/

PLEASE refer to CasADi's documentation for all details!

Author: Pedro Roque, padr@kth.se
"""

from casadi import *
import matplotlib.pyplot as plt

# SX variables - TL;DR: matrices with elements that are sequences unary or binary operations
"""
The SX data type is used to represent matrices whose elements consist
of symbolic expressions made up by a sequence of unary and binary operations.
"""
xs = SX.sym("x")
ys = SX.sym("y", 10, 2)

# MX variables - TL;DR: like SX, but elements are not limited to unary or binary operations
# Note: SX variables often result in a better performance when solving big optimization problems,
#       compared to MX variables
"""
The MX type allows, like SX, to build up expressions consisting of a sequence
of elementary operations. But unlike SX, these elementary operations are not
restricted to be scalar unary or binary operations. Instead, the elementary
operations that are used to form MX expressions are allowed to be general
multiple sparse-matrix valued input, multiple sparse-matrix valued output functions.
"""
xm = MX.sym("x")
ym = MX.sym("y", 10, 2)


# DM variables - TL;DR: represents all purely numeric variables
"""
DM is very similar to SX, but with the difference that the nonzero elements are
numerical values and not symbolic expressions. The syntax is also the same, except
for functions such as SX.sym, which have no equivalents.
"""
x = DM.ones(2, 1)
A = DM.eye(2)

# Transposition
B = DM([[2, 1], [4, 3]])
C = B.T

# Multiplications
v = mtimes(A, x)            # Matrix product
v = mtimes([x.T, A, x])     # Matrix product
v = A * A                   # Element-wise product

# Concatenation
vc = vertcat(A, B)
hc = horzcat(A, B)

# Reshaping
column_matrix = vec(A)
reshaped_matrix = reshape(A, (1, 4))

# Slicing
B[0, 0]                     # Take entry 0,0 of B (first entry)
B[:, 0]                     # Take the first column of B
B[-1, :]                    # Take the last line of B

# Functions - scalar input, scalar output
x = xs**2
f = Function('f', [xs], [x])
x_res = f(2)

# Function - two scalar inputs, one output
x = xs**2 + ys**2
f = Function('f', [xs, ys], [x])
x_res = f(2, 3)

# Function - vector input, scalar output
vs    = SX.sym('vs', 3, 1)
x     = norm_1(vs)
f     = Function('f', [vs], [x])
x_res = f(DM.ones((3, 1)))

# Function - two inputs, two outputs
As = SX.sym('As', 3, 3)
x = norm_1(vs)
y = As[-1, :]
f = Function('f', [vs, As], [x, y])
x_res, y_res = f(DM.ones((3, 1)), DM.eye(3))

# Functions - named inputs/outputs
f = Function('f', [vs, As], [x, y],
             ['vs', 'As'], ['x', 'y'])
res = f(vs=DM.ones((3, 1)), As=DM.eye(3))
print(res['x'], res['y'])

# Jacobian of a function
J = jacobian(sin(xs), xs)
print(J)                # cos(x)

# Hessian
x2 = SX.sym('x2', 2)
H, g = hessian(dot(x2, x2), x2)
plt.spy(H.sparsity())
plt.show()              # Shows the sparsity pattern of H

# Quadric Programming
zs = SX.sym('zs')
y = xs**2 + zs**2
solver = qpsol('solver', 'qpoases', {'x': vertcat(xs, zs), 'f': y})
res = solver(**{'x0': vertcat(0.1, 0.2)})
print("Minimizer: ", res['x'])              # Minimum
print("Minimum: ", res['f'])                # Objective value

# Nonlinear Programming
solver = nlpsol('solver', 'ipopt', {'x': vertcat(xs, zs), 'f': y})
res = solver(**{'x0': vertcat(0.1, 0.2)})
print("Minimizer: ", res['x'])              # Minimum
print("Minimum: ", res['f'])                # Objective value

# Integrator
# Take the system:
#  dot(x1) = 1
#  dot(x2) = x1**2 + x2**2
x1 = SX.sym('x1')
x2 = SX.sym('x2')
y = x1**2 + x2**2
intg = integrator('intg', 'cvodes', {'x': vertcat(x1, x2), 'ode': vertcat(1, y)}, {"tf": 0.1})
res = intg(x0=DM.ones(2, 1))
print("State after 'tf': ", res['xf'])      # key 'xf' contains the value of the x at the final timestamp 'tf'

# Integrator - system with external input
# Take the system:
#  dot(x1) = 1
#  dot(x2) = x1**2 + x2**2 + u
us = SX.sym('u')
y = x1**2 + x2**2 + us
intg = integrator('intg', 'cvodes', {'x': vertcat(x1, x2), 'ode': vertcat(1, y), 'p': us}, {"tf": 0.1})
u = 1
res = intg(x0=DM.ones(2, 1), p=u)
print("State after 'tf' with input: ", res['xf'])

print("\n-----\n\n  If you understood it all, you are ready!\n\n-----")
