# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 

'''
Original run time:1.74 s
Current run time:14.2 ms
Relative speedup:122.5
Process:
By running profile module, I found that sqrt function is the most-time consuming function all.
Then, by running line_profiler, I found the double for loops in sqrt function costs most.
However, there was no way to avoid using double for loop if we wrote the python code ourselves.
Thus,I substituted the sqrt code with the built-in functions in numpy.
Then I repeated the above steps until add, multiply, and sqrt are all optimized.
hypotenuse function ran very quick,so I did not optimize hypotenuse function.
'''
import numpy as np

def add(x,y):
    """
    Add two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    return np.add(x,y)


def multiply(x,y):
    """
    Multiply two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    return np.multiply(x,y)


def sqrt(x):
    """
    Take the square root of the elements of an arrays using a Python loop.
    """
    return np.sqrt(x)


def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = multiply(x,x)
    yy = multiply(y,y)
    zz = add(xx, yy)
    return sqrt(zz)
