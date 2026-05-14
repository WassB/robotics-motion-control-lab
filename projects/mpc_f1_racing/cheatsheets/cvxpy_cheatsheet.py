"""
CVXPY Cheat Sheet - made for EL2700

CVXPY Documentation:
https://www.cvxpy.org/index.html

CVXPY Tutorials :
https://www.cvxgrp.org/cvx_short_course/docs/index.html


PLEASE refer to CVXPY's documentation for all details!
All rights reserved to CVXPY's authors for the examples proposed.

Author: Gregorio Marchesini, padr@kth.se
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

######### VARIABLES AND PARAMETERS ###################
# Define variables 
x = cp.Variable(2)
y = cp.Variable(2)

# Define parameters 
theta  = cp.Parameter(nonneg=True) # scalar parameter
A      = cp.Parameter((2, 2)) # matrix parameter
S      = cp.Parameter((2, 2), PSD=True) # symmetric positive semidefinite matrix parameter

# Define constants 
B  = np.array([[1, 2], [3, 4]]) # matrix
C  = np.random.randn(2, 2)      # matrix
vc = np.array([[1], [2]])       # column vector
vr = np.array([1, 2])           # row vector
a = 3

""" 
CVXPY is highly compatible with Numpy so that most of the 
operations you have available in Mumpy are also available in CVXPY
"""

###### BASIC OPERATIONS AND EXPRESSIONS #########
# Basic operations expressions
expr_1 = x + y
print("only variables expressions : ",expr_1)

expr_2 = A @ x + B @ y
print("variables,parameters and constants expressions : ",expr_2)

# Transposition
print(A)
print(A.T)

# Multiplications
print("Matrix-vector multiplication : ", A@y) # @ should be used for matrix-matrix and matrix-vector multiplication,
print("Matrix-scalar multiplication : ", 3*A) # * should be used for scalar multiplication
print("Element wise multiplication  : ", cp.multiply(A,B)) # element wise multiplication
# try this : print(A * B), you should see a warning appearing !


# # Concatenation
ABc = cp.vstack([A, B])
print("Vertical concatenation : ", ABc)
ABr = cp.hstack([A, B])
print("Horizontal concatenation : ", ABr)

# Reshaping
row_matrix   = cp.vec(A) 
print("Flattened matrix : ", row_matrix)
col_matrix   = cp.reshape(A, (4, 1))
print("Column matrix : ", col_matrix )
vr_after           = np.expand_dims(vr, axis=1)
print("Before expansion : ", vr)
print("After Expansion   : ", vr_after)

# Slicing
B[0, 0]                     # Take entry 0,0 of B (first entry)
B[:, 0]                     # Take the first column of B
B[-1, :]                    # Take the last line of B

###### FUNCTIONS #########
"""
Just a a couple of functions example but many more can be found here 
https://www.cvxpy.org/tutorial/functions/index.html#operators:~:text=on%20the%20approximations.-,Vector/matrix%20functions,-%C2%B6
"""
# norm squared 
norm_squared = cp.sum_squares(x) # or x.T@x ( avoid cp.norm(x)**2 )

# quadratic form
quad_form = cp.quad_form(x, S) # aka x.T@S@x
# Try : quad_form = cp.quad_form(x, A) # why do you think an error occurs with the matrix A?


# clearing previous variables to prepare for next section
del x,y,theta,A ,S


############## CONVEX OPTIMIZATION WITH CVXPY ###################
""" 
CVXPY can be used to minimize `convex functions over convex sets`.
CVXPY will enforce such conditons and it will check that constraints and objectives
for the optimization program respect the convexity rules. 

Namely the following rules must be respected :
- The objective function must be convex 
   1) Minimize(Convex Objective)
   2) Maximimize(Concave Objective) (equivalent to  -1* Minimize(-1*Concave Objective))

- The constraints must be convex !
    affine == affine
    convex <= concave
    concave >= convex
"""


#  Linear Programming
print("---------Linear Programming Example-----------")
x = cp.Variable()
y = cp.Variable()

cost        = x + y # affine cost
constraints = [10*x + 2*y >=10,
              -10*x  -2*y <=30,
              -2*x  +10*y <=30,
               2*x  -10*y <=30]          # affine constraint
objective   = cp.Minimize(cost)     # create objective 
prob        = cp.Problem(objective, constraints) # create the problem
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("Is the problem in Disiplined Convex Programming form : ", prob.is_dcp())
print("Value of x : ", x.value)
print("Value of y : ", y.value)

#  Quadratic Programming
print("---------Quadratic Programming-----------")
x = cp.Variable()
y = cp.Variable()

cost        = x**2 + y**2           # convex cost
constraints = [x + y == 1]          # affine constraint
objective   = cp.Minimize(cost)     # create objective 
prob        = cp.Problem(objective, constraints) # create the problem
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("Is the problem in Disiplined Convex Programming form : ", prob.is_dcp())
print("Value of x : ", x.value)
print("Value of y : ", y.value)


# Parametric quadratic programming 
print("---------Parametric quadratic programming -----------")
from matplotlib.patches import Ellipse,Rectangle
"""
We will find the rectangle that best fits in a given ellipse
"""
del x,y

height = cp.Variable(pos=True) # height of the rectangle (implicitly sets the constraint height >= 0)
width  = cp.Variable(pos=True) # width of the rectangle  (implicitly sets the constraint width >= 0 so you don't have to do it yourslef)
P      = cp.Parameter((2, 2), PSD=True) # matrix parameter defyning the ellipse via the equation x.T@P@x <= 1

# define vertices of rectangle as a function of the parameters 
v1 = cp.vstack([-width/2, -height/2])
v2 = cp.vstack([-width/2, height/2])
v3 = cp.vstack([width/2, height/2])
v4 = cp.vstack([width/2, -height/2])
vertices = [v1, v2, v3, v4]

# now we impose that each vertex of the rectangle should be included in the ellipse
constraints = [cp.quad_form(v,P) <= 1 for v in vertices] # equivalent to v.T@P@v <= 1
# impose also positivity of the width and height constraints += [width >= 0, height >= 0]

# We want to maximize the volume of the rectangle given by volume = width*height
# One way to maximize the volume is to minimize the inverse of the volume a.k.a Minimize(1/volume)

cost        = -(width + height)  # we add a small constant to avoid division by zero
objective   = cp.Minimize(cost)     # create objective
prob        = cp.Problem(objective, constraints) # create the problem

semimajoraxes_values = [(2,3),(1,1),(8,5)]
fig,ax = plt.subplots()

for semimajoraxes in semimajoraxes_values:
    # set the value of the P
    P.value = np.array([[1/semimajoraxes[0]**2, 0], [0, 1/semimajoraxes[1]**2]]) # ellipse
    print(cost.is_dcp())
    prob.solve()

    print("status:", prob.status)
    print("optimal value", prob.value)
    print("Is the problem in Disiplined Convex Programming form : ", prob.is_dcp())
    print("Value of x : ", height.value)
    print("Value of y : ", width.value)
    


    ellipse = Ellipse(xy=(0,0), width=semimajoraxes[0]*2, height=semimajoraxes[1]*2, 
                        edgecolor='r', fc='None', lw=2)
    rectangle = Rectangle(xy=(-width.value/2,-height.value/2), width=width.value, height=height.value,fc='None', lw=2, edgecolor='b')
    ax.add_patch(ellipse)
    ax.add_patch(rectangle)

ax.set_xlim(-8,8)
ax.set_ylim(-8,8)
plt.show()

"""
You should have received the following warnign message from the previous example :

`UserWarning: You are solving a parameterized problem that is not DPP. Because the problem is not DPP, subsequent solves will not be faster than the first one.
For more information, see the documentation on Disciplined Parametrized Programming, at https://www.cvxpy.org/tutorial/dpp/index.html`

The reason why you see the message it is because the matrix P since tha constraint cp.quad_form(v,P) is not affine in P. CVXPY only 
can efficently parameterize problems for which the parameters enter affinely in the defintion of the constraint/objective. This is explained carefully here 
https://www.cvxpy.org/tutorial/dpp/index.html

We will see during the course  that the MPC problem can actually be written as a DPP problem so that we can exploit the efficieny of the parametrization
"""

