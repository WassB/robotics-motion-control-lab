import numpy as np
import matplotlib.pyplot as plt

"""
This script is meant to refresh your knowledge in Python, Numpy.

The first part of the script is taken from the CVXPY tutorials webape which you are welcome to visit here :
https://www.cvxgrp.org/cvx_short_course/docs/python_intro/notebooks/numpy_example.html

The second part of the script is dedicated to solving a few simple exercises 

Pedro Roque, Gregorio Marchesini 
Last update : 2024/06
"""

#######################################################
#######################################################
#                      PART -1
#  Basic types, Lists, Tuples, Dictionaries, and Sets
#######################################################
#######################################################
# Python is an interpreted, object-oriented, high-level programming language.
# The main building blocks of a Python program are variables (of a specific data type),
# and functions (which are blocks of code that perform a specific task over the variables).
# On top of this, python is rich in data structures such as lists, tuples, dictionaries, and sets,
# that can be used to group and order multiple variables.
# The next section is dedicated to give a brief refresher types, data structures and functions is python

# Basic types
#   int, float, str, bool
a = 1
b = 2.0
c = "hello"
d = True


# operations over types
print(a + b)  # Addition
print(a * b)  # Multiplication
print(a / b)  # Division
print(a ** b)  # Exponentiation
print(a % b)  # Modulus
print(a == b)  # Equality
print(a != b)  # Inequality
print(a > b)  # Greater than
print(a < b)  # Less than
print(a >= b)  # Greater than or equal to
print(a <= b)  # Less than or equal to

# string operatiions 
print(c + " world")  # Concatenation
print(c * 3)  # Repetition
print(c[0])  # Indexing (0-indexed)
print(c[1:3])  # Slicing (1-indexed to 2-indexed)
print(len(c))  # Length


# Lists
#   Ordered, mutable, allows duplicates
L = [1, 2, 3, 4, 5]
print(L)
print(L[0])  # First element (0-indexed)
print(L[-1])  # Last element
print(L[1:3])  # Slicing (1-indexed to 2-indexed)
L.append(6)  # Append an element
print(L)    

# Note for matlab users: Python is 0-indexed, meaning that the first element of a list is indexed by 0
# and the last element is indexed by len(list) - 1. So if you want to access the first element of a list
# you should use list[0] and if you want to access the last element you should use list[- 1].
# Moreover ranges in python are defined as [start:stop:step] where start is included and stop is NOT included.
# Indeed L[:3] can be used to access the first 3 elements of a list L, but the last element is the one at index 2.
# Indeed counting from 0 to 2 gives 3 elements ! (0,1,2).

# tuples
#   Ordered, immutable, allows duplicates
T = (1, 2, 3, 4, 5)
print(T)
print(T[0])  # First element (0-indexed)
print(T[-1])  # Last element
print(T[1:3])  # Slicing (1-indexed to 2-indexed)

# Note for matlab users: Tuples are similar to lists but they are immutable, meaning that once they are created
# they cannot be modified. This is useful when you want to create a list of elements that should not be modified
# during the execution of the program.

# Dictionaries
#   Unordered, mutable, indexed, no duplicates
D = {"a": 1, "b": 2, "c": 3}
print(D)
print(D["a"])  # Access value by key
D["d"] = 4  # Add a new key-value pair
print(D)
D.get("e", 5)  # Get value by key, with default of 5 if key not found

# Note for matlab users: Dictionaries are a way to store key-value pairs. This is useful when you want to store
# information that can be accessed by a specific key. For example, if you want to store the age of a person
# you can create a dictionary with the name of the person as the key and the age as the value. This way you can
# access the age of a person by using the name of the person as the key.


# Sets
#   Unordered, mutable, no duplicates
S = {1, 2, 3, 4, 5}
print(S)
S.add(6)  # Add an element
print(S)
S1 = {1, 2, 3}
S2 = {3, 4, 5}
print(S1.union(S2))  # Union
print(S1.intersection(S2))  # Intersection

# Note sets are useful when you want to store a collection of elements that should not be repeated.
# For example, if you want to store the unique elements of a list you can convert the list to a set.
# Example : L = [1, 2, 3, 1, 2, 3] -> S = set(L) -> S = {1, 2, 3}


# Functions
# Functions are main building blocks of a Python program. They are blocks of code that perform a specific task.

# adding two numbers
def my_function(a, b):
    return a + b

# calling the function
print(my_function(1, 2))

# create a function to read the keys of a dictionary 
def read_keys(D):
    for key in D.keys():
        print(key)
    return 

# calling the function
print(read_keys(D))

# A function can return multiple values
def my_function(a, b):
    return a + b, a - b

# calling the function
c, d = my_function(1, 2) # the comma can be used to unpack objects in python
print(c, d)



# Objects and Classes
# Python is an object-oriented programming language. This means that it allows the definition of classes and objects.
# in reality, everything is an object is python like the types we saw before (int, float, str, bool, list, tuple, dict, set)
# but you can also create your own objects !.

# Example of a class
class Person:
    
    # every object of a class has an __init__ method that is called when the object is created
    def __init__(self, name : str, age :float):
        self.name = name  # this is an attribute of the object
        self.age = age    # this is another one
    
    # a function within an an object is commonly called a method
    def print_info(self):
        print("Name: ", self.name)
        print("Age: ", self.age)
        
# creating an object of the class Person
p = Person(name = "Greg", age= 25) # you can also call simply as Person("Greg", 25) but you need to remember the order in that case!
p.print_info()

# Note that the __init__ method is called when the object is created and it is used to initialize the object.
# The self argument is a reference to the object itself and it is used to access the attributes and methods of the object.
# The attributes of the object are defined within the __init__ method and they are accessed using the self argument.
# The methods of the object are defined within the class and they are accessed using the self argument as well.

# You could also overload basic operators in python for your class (this is why int,float and other types can be added, multiplied, etc)

# Example of a class with overloaded operators
class Vector:
        
        def __init__(self, x : float, y : float):
            self.x = x
            self.y = y
        
        # addition between two vectors
        def __add__(self, other):
            return Vector(self.x + other.x, self.y + other.y)
        
        def __mul__(self, other):
            return self.x * other.x + self.y * other.y
        
        # used when the print function is called on the object
        def __str__(self):
            return "x: " + str(self.x) + " y: " + str(self.y)
        
# example of usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2
print(v3)
print(v1 * v2)
# print(v1/v2) # this will raise an error because the division operator is not defined for the class Vector

# Some final notes:
# Note 1: functions in python are also objects ! you can pass them as arguments to other functions, return them from functions, etc.
# Note 2: Python is a dynamically typed language, meaning that you do not need to specify the type of a variable when you declare it (like in matlab).
# Note 3: it is always a good practice to comment your code and introduce a funciton signature to your functions if the function is not trivial. As a certainty in your life, you will forget what you did in a few months.
# Note 4: This is just an intro to python. There is waaaaaaaay more to it to be learned and python is notoriously the most used language worldwide, so it is worth learning it


# #######################################################
# #######################################################
# #                      PART 0
# #######################################################
# #######################################################

# In this section we revise how to use while loops, for loops, if statements, and list comprehensions in Python.
# these are core functionalities of python that you will use a lot in your programming tasks.

# While loops
#   Execute a block of code while a condition is true
i = 0
while i < 5:
    print(i)
    i += 1

# note in Python (differently than in Matlab) the indentation is very important. The code inside the while loop is indented and the indentantion
# is used to define the block of code that is executed in the while loop. This is a way to make the code more readable and to avoid the use of brackets.
# There is NO END statement in python to close a block of code. The end of the block is defined by the indentation. Python will throw an error if you miss 
# the indentation.

# for loops 
#   Execute a block of code for each element in a sequence
L = [1, 2, 3, 4, 5]

for i in L:
    print(i)
    
# in matlab we are used to iterate over indices of a list. In python you can do this as well
for i in range(len(L)):
    print(L[i])
    
# but you should not do it ! it is not pythonic. The pythonic way to iterate over a list is to iterate over the elements of the list directly as we did before.
# look at at how cool is the syntax in python using this functionality 

people = ["Greg", "Pedro", "John", "Alice"]
for person in people:
    print(person)

# in gereal you can iterate on everything that is an instance of the class Iterable. This includes lists, tuples, dictionaries, sets, etc.
# here are some example 
set1 = {1, 2, 3, 4, 5}
for number in set1:
    print(number)
    
dict1 = {"a": 1, "b": 2, "c": 3}
for key in dict1.keys():
    print(key)
# dictionaries also have a way to iterate over both keys and values (THIS IS EXTREEEEEEMLY USEFUL AND USED A LOT)
for key, value in dict1.items():
    print(key, value)

# finally you can use ranges to iterate over a sequence of numbers
for i in range(5):
    print(i)
# the class enumerate is also very useful to iterate over a list and get the index of the element

for i, person in enumerate(people):
    print(i, person)
    


# if statements
#   Execute a block of code if a condition is true
a = 1
if a == 1:
    print("a is 1")
elif a == 2:
    print("a is 2")
else:
    print("a is neither 1 nor 2")
    
# list comprehensions
# This is nothing like in matlab. This is a very powerful feature of python that allows you to create lists in a very concise way.

numbered_people = [(i, person) for i, person in enumerate(people)] # create a list of tuple with the index and the person

# create a dictionary with list comprehensions
people = ["Greg", "Pedro", "John", "Alice"]
ages_dict  = {person : 25 for person in people} # create a dictionary with the age set to 25 of each person

# tricks : the zip method
# zip is a very useful method that allows you to iterate over two lists at the same time (it pairs every element of the two lists into a tuple)

ages = [25, 30, 35, 40]
for person, age in zip(people, ages):
    print(person, age)
    
    

# #######################################################
# #######################################################
# #                      PART 1
# #######################################################
# #######################################################

# In this section you will learn how to use numpy, a library that is widely used in python for numerical computations.
# Numpy is different from matlab in how arrays work. We must be honest sayiing that matlab was created for the easiness in matrix and vector operations so
# that using matlab for vector an matrix operations is often easier. But a bit of practice will make you feel comfortable with numpy as well.  


# 1: Create a vector
v = np.array([1, 2, 3, 4, 5])
print(v)
# an array in numpy ha only one dimension! in matlab a vector is a 1xN or Nx1 matrix
# in numpy a vector is a 1D array. If you want to create a 2D array you should use np.array([[1, 2, 3, 4, 5]])
# note indeed what happens try to transpose the vector 
print(v.T) # this will not work as expected
print(v)

# The array is the same as before! this is because the array is 1D and the transpose of a 1D array is the same array.
# Usually in python it is best to think of  a vector as a row and not as a column like in matlab.

# 2: Create a column vector
v = np.array([[1], [2], [3], [4], [5]])
print(v)
print(v.T) # now the transpose works as expected because the vector has two dimensions 

# 3: a trick to make a vector a column vector
v = np.array([1, 2, 3, 4, 5])
v = v[:, np.newaxis] # this command adds one dimension to the vector
print(v)
print(v.T) # now the transpose works as expected because the vector has two dimensions

# 4: Create a matrix
#   NOTE: argument is a tuple '(3, 4)'
#     WRONG: np.zeros(3, 4)
#     CORRECT: np.zeros( (3, 4) )
A = np.zeros((3, 4))

# you can create also a tensor of zeros
T = np.zeros((3, 4, 5))


print(A)
print(A.shape)  # dimensions of A

# All-one matrix
B = np.ones((3, 4))
print(B)

# Identity matrix
I = np.eye(5)
print(I)

# Stacking matrices horizontally
#   Use vstack to stack vertically
J = np.hstack((I, I))
print(J)

# Random matrix with standard Gaussian entries
#   NOTE: argument is NOT a tuple
Q = np.random.randn(4, 4)

print(Q)
print(Q[:, 1])  # Second column (everything is 0-indexed)
print(Q[2, 3])  # (3, 4) entry (as a real number)


# Random column vector of length 4
v = np.random.randn(4, 1)

# v.T: v tranpose
# @: matrix multiplication (it only works if you have a 2D arrays)
z = v.T @ Q @ v

# The result is a 1-by-1 matrix
print(z)

# Extract the result as a real number
print(z[0, 0])

# Other useful methods
#   Construct a matrix
A = np.array([[1, 2], [3, 4]])
B = np.array([[-1, 3.2], [5, 8]])
#   Transpose a matrix
print(A.T)
#   Elementwise multiplication (IT IS NOT THE SAME AS MATRIX MULTIPLICATION)
print(np.multiply(A, B))
#   Sum of each column (as a row vector)
print(np.sum(A, axis=0))
#   Sum of each row (as a column vector)
print(np.sum(A, axis=1))

# Linear algebra routines
Q = A.T @ A
(d, V) = np.linalg.eig(Q)  # Eigen decomposition
print("d = ", d)
print("V = ", V)

v = np.array([1, 2])
print("||v||_2 = ", np.linalg.norm(v))  # 2-norm of a vector

Qinv = np.linalg.inv(Q)  # Matrix inverse
# Solves Qx = v (faster than Qinv*v)
x = np.linalg.solve(Q, v)
print("Q^{-1}v = ", x)

############### EXTREEEMLY IMPORTANT IF YOU ARE MATLAB USER ###########################
# One very important concept in Numpy regards how vectors/matrices are stored in the memory.
# This is often cause of mistakes that are difficult to debug.
# Namely, numpy array are not copied unless you ask numpy to do so.

# consider 
v1 = np.array([1, 2, 3])
print("The vectors a looks like :", v1)
v2 = v1
print("The vectors a looks like v1 clearly:", v2)
# now if you change v1, also v2 will be changed ! this is because v2 is a reference to the same memory allocation
v1[2] = 5
print("After changing v1 look at v2! :", v2)
# numpy offers the possibility to instead create a copy of the array
# so now the two copies are independent
v2 = v1.copy()
v1[2] = 7
print("Now the copies are independent :", v2)
# the reason for this is that numpy is designed to be very efficient in memory usage. If you have a very large array and you copy it, you will
# use twice the memory. This is why numpy does not copy arrays by default. But you should be aware of this behaviour to avoid bugs in your code!


#######################################################
#######################################################
#      PART 2 : some exercises
#######################################################
#######################################################


############ Q1 ###################
# Consider the following matrix A 
A = np.zeros((9, 9))

# Q1.1 : Knowing that in Python the iterator format is:
#        start:stop:step , so 1::2 means start at 1, stop at the end
#                   and use a step of 2 -> 1, 3, 5, ...
#         Change the matrix A so that the border of A is 0 but the core
#         of A follows the pattern in:
#                       [0, 0, 0, 0, 0]
#                       [0, 1, 0, 1, 0]
#                       [0, 0, 0, 0, 0]
#                       [0, 1, 0, 1, 0]
#                       [0, 0, 0, 0, 0]

############ Q2 ###################
# Take the following matrix A and array b
A = np.eye(3) * 10
b = np.linspace(1, 3, 3)

# Q2.1 : What is the shape of A and b? (hint : use the .shape method)
# Q2.2 : Calculate and print
#       i) A @ b
#       ii) b^T @ A @ b
#       iii) A^{-1} @ b
#       iv) || b || (norm-2)
#       (hint 1: the symbol @ indicates matrix multiplication in Numpy and it is NOT equivalent to * which indicates element wise multiplication. 
#       This is different from Matlab where * is matrix multiplication and .* is element wise multiplication.)
#       (hint 2: the module np.linalg contains useful functions for linear algebra operations such as inv, norm, etc.)

############ Q3 ###################
# Q3.1 : With np.linspace or np.arange and np.sin, plot a sine function
#        using plt.plot(x, y) and plt.show()


############ Q4 ###################
# Q4.1 : Take now a random array such as
b_rnd = np.random.uniform(low=0, high=10, size=(1001))
# Get the 2nd largest value in the array by creating a for loop
# that looks for this item and stores its index too.
# Useful functions: range(), len()



#######################################################
#######################################################
#      PART 3 : plotting
#######################################################
#######################################################

# the library matplotlib is used to plot data in python
# it is very similar to matlab in the way it is used

# create a vector x
x = np.linspace(0, 2*np.pi, 100)
# create a vector y
y = np.sin(x)

# plot the data
fig,ax = plt.subplots()
ax.plot(x, y)
ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')

# create scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
fig,ax = plt.subplots()
ax.scatter(x, y)
ax.grid()

# create subplots
fig, axs = plt.subplots(2, 2)
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x)

axs[0, 0].plot(x, y1)
axs[0, 1].plot(x, y2)
axs[1, 0].plot(x, y3)
axs[1, 1].plot(x, y4)

axs[0, 0].set_title('sin(x)')
axs[0, 1].set_title('cos(x)')
axs[1, 0].set_title('tan(x)')
axs[1, 1].set_title('exp(x)')

# grids
for ax in axs.flat:
    ax.grid()




plt.show()# if you don't put this command the plot will not be shown in most of the editors