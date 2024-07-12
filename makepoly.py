#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:55:47 2024

@author: Caleb Erickson
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math
import warnings


'''
makepoly: this function creates a polynomial out of a list of x values, y values and the degree of the polynomial
========================================================================
input   xn: this is the list of x values
        yn: this is the list of y values
        N: this is the degree of the polynomial
output returns: list of polynomial coefficients
'''
def makepoly(xn, yn, N):
    #create a list with N+1 zeros
    a = np.zeros(N+1)
    #get the number of x values inputed
    n = len(xn)
    #if the number of inputs is greater than the degree+1
    if n>N+1:
        warnings.warn(f"Requested polynomial of degree {N}, but provided {n} nodes.", RuntimeWarning)
    #if the number of inputs is less than the degree+1
    elif n<N+1:
        raise ValueError(f"Requested a polynomial of degree {N} but only provided {n} nodes.")
    
    #now set n to the nmber of y value inputed
    n = len(yn)
    #if the number of inputs is greater than the degree+1
    if n>N+1:
        warnings.warn(f"Requested polynomial of degree {N}, but provided {n} y-values for nodes.", RuntimeWarning)
    #if the number of inputs is less than the degree+1
    elif n<N+1:
        raise ValueError(f"Requested a polynomial of degree {N} but only provided {n} y-values for nodes.")
    
    #set the initial value for array:   a_0 = yn_0
    a[0] = yn[0]
    for k in range(1,N+1):
        w = 1.
        p = 0
        for j in range(k):
            p = p + a[j]*w
            w = w * (xn[k]-xn[j])
        a[k] = (yn[k] - p)/w
        
    return a
   


'''
evalpoly: This evaluates a polynomial, with a given range, a given set of x values, a polynomial function, and the degree of the polynomial
=====================================================================
input   x: This is the range of x values?
        xn: this is a list of x values, for which we want to compute f(x)
        a: This is the function of the polynomial. i.e. f(x)
        N: This is the degree of the polynomial=(nodes+1)

output returns a list of y values
'''
def evalpoly(x, xn, a, N):
    #
    n = len(xn)
    if n>N+1:
        warnings.warn(f"Requested evaluation of polynomial of degree {N}, but provided {n} nodes.", RuntimeWarning)
    elif n<N+1:
        raise ValueError(f"Requested a polynomial of degree {N} but only provided {n} nodes.")
    
    n = len(a)
    if n>N+1:
        warnings.warn(f"Requested evaluation of polynomial of degree {N}, but provided {n} coefficients.", RuntimeWarning)
    elif n<N+1:
        raise ValueError(f"Requested a polynomial of degree {N} but only provided {n} coefficients.")
    
    px = a[N]
    for k in range(N-1,-1,-1):
        xd = x - xn[k]
        px = a[k] + px*xd
    return px
 


"""
Part 2: Tests for interpotion
Three tests with-hand calculated results
"""

eps = 1e-15 

xn = np.array([-1,1,3])
yn = np.array([0,2,1])
ahand = np.array([0,1,-3/8])
a= makepoly(xn,yn,2)

'''
check hand calculations vs. makepoly
'''
assert(all(abs(a-ahand)<eps))
'''
check that nodes are interpolated correctly
'''
y2 = evalpoly(xn,xn,a,2)
assert(all(abs(yn-y2)< eps))


#print(makepoly([-1,1,3], [0,2,1], 2)) # this coincides with my handwork of:  0 + (x+1) - (3/8)(x+1)(x-1)


xnb = np.array([0,1,2,-1])
ynb = np.array([1,0,-3,-6])
bhand = np.array([1,-1,-1,1])
b = makepoly(xnb, ynb, 3)

'''
check hand calculations vs. makepoly
'''
assert(all(abs(b-bhand)<eps))
'''
check that nodes are interpolated correctly
'''
y2b = evalpoly(xnb,xnb,b,3)
assert(all(abs(ynb-y2b)< eps))


#print(makepoly([0,1,2,-1], [1,0,-3,-6], 3)) #this coincides with my handwork of: 1 - (x) - (x)(x-1) + (x)(x-1)(x-2)


xnc = np.array([0,-1,1,2,3])
ync = np.array([-1,2,0,5,-10])
chand = np.array([-1,-3,2,0,-1])
c = makepoly(xnc, ync, 4)

'''
check hand calculations vs. makepoly
'''
assert(all(abs(c-chand)<eps))
'''
check that nodes are interpolated correctly
'''
y2c = evalpoly(xnc,xnc,c,4)
assert(all(abs(ync-y2c)< eps))

print(makepoly([0,-1,1,2,3], [-1,2,0,5,-10], 4)) #this coincides with my handwork of: -1 - 3(x) + 2(x)(x+1) - (x)(x+1)(x-1)(x-2)


"""
Part 3: steam pressure example
"""
#complete list of data points from the book
T = [220, 230, 240, 250, 260, 270, 280, 290, 300]
P = [17.188, 20.78, 24.97, 29.82, 35.42, 41.85, 49.18, 57.53, 66.98]


#3a
T1 = [T[0], T[4], T[len(T)-1]]
P1 = [P[0], P[4], P[len(P)-1]]

#print(T1)
#print(P1)

A3 = makepoly(T1, P1, 2)

#print(A3)

"""
First plot /// Quadratic
"""
# Generating points to plot the polynomial
x_values = np.linspace(180,350,200)
y_values = evalpoly(x_values, T1, A3, 2)

# Plotting the polynomial and the data points
plt.plot(T, P, 'bo', label='Data')
plt.plot(x_values, y_values, 'r-', label='Polynomial')

# Adding labels and legend
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('Degree 2 Interpolation')
plt.legend()

plt.savefig('InterpolationDeg2.png',dpi =200) #uncomment to save graph
# Showing the plot
plt.show()
plt.close()



#3b
T2 = [T[0], T[2], T[4], T[6], T[len(T)-1]]
P2 = [P[0], P[2], P[4], P[6], P[len(P)-1]]

B3 = makepoly(T2, P2, 4)

"""
2nd plot /// Quartic 
"""
# Generating points to plot the polynomial
x_values = np.linspace(180,350,200)
y_values = evalpoly(x_values, T2, B3, 4)

# Plotting the polynomial and the data points
plt.plot(T, P, 'bo', label='Data')
plt.plot(x_values, y_values, 'g-', label='Polynomial')

# Adding labels and legend
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('Degree 4 Interpolation')
plt.legend()

plt.savefig('InterpolationDeg4.png',dpi =200) #uncomment to save graph
# Showing the plot
plt.show()
plt.close()



"""
Both on the same graph
"""
x_values = np.linspace(180,350,200)
y_values_A3 = evalpoly(x_values, T1, A3, 2)
y_values_B3 = evalpoly(x_values, T2, B3, 4)

# Plotting the polynomials and the data points
plt.plot(T, P, 'bo', label='Data')
plt.plot(x_values, y_values_A3, 'r-', label='Quadratic Interpolation')
plt.plot(x_values, y_values_B3, 'g-', label='Quartic Interpolation')

# Adding labels and legend
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('Quadratic Interpolation and Quartic Interpolation graphed with actual data points')
plt.legend()

plt.savefig('QuadraticVsQuartic.png',dpi =200) #uncomment to save graph
# Showing the plot
plt.show()
plt.close()




"""
Graphing the relative errors of A3 and B3 to the actual data T and P
"""
x_values = np.linspace(180,350,200)
actual_pressures = np.interp(x_values, T, P)
interpolated_pressures_A3 = evalpoly(x_values, T1, A3, 2)
interpolated_pressures_B3 = evalpoly(x_values, T2, B3, 4)
relative_error_A3 = np.abs((interpolated_pressures_A3 - actual_pressures) / actual_pressures)
relative_error_B3 = np.abs((interpolated_pressures_B3 - actual_pressures) / actual_pressures)

# Plotting the relative errors
plt.plot(x_values, relative_error_A3, label='Relative Error Quadratic Interpolation')
plt.plot(x_values, relative_error_B3, label='Relative Error Quartic Interpolation')

# Adding labels and legend
plt.xlabel('Temperature')
plt.ylabel('Relative Error')
plt.title('Graphing the Relative errors of previous 2 graphs')
plt.legend()

plt.savefig('RelErrorQuarVsQuad.png',dpi =200) #uncomment to save graph
# Showing the plot
plt.show()
plt.close()




# Relative errors at data points
interpolated_pressures_A3_nodes = [evalpoly(t, T1, A3, 2) for t in T]
interpolated_pressures_B3_nodes = [evalpoly(t, T2, B3, 4) for t in T]
relative_error_A3_nodes = np.abs((np.array(interpolated_pressures_A3_nodes) - np.array(P)) / np.array(P))
relative_error_B3_nodes = np.abs((np.array(interpolated_pressures_B3_nodes) - np.array(P)) / np.array(P))

# Scatter plot of the relative errors at the nodes
plt.scatter(T, relative_error_A3_nodes, label='Relative Error Quadratic Interpolation', color='red')
plt.scatter(T, relative_error_B3_nodes, label='Relative Error Quartic Interpolation', color='green')
plt.xlabel('Temperature')
plt.ylabel('Relative Error')
plt.title('Relative Errors at Data Points')
plt.legend()
plt.savefig('RelativeErrorsAtNodes.png', dpi=200)
plt.show()
plt.close()






















"""
Now plotting a degree 8 polynomial
"""
D3 = makepoly(T, P, 8)
# Generating points to plot the polynomial
x_values = np.linspace(180,350,200)
y_values = evalpoly(x_values, T, D3, 8)

# Plotting the polynomial and the data points
plt.plot(T, P, 'bo', label='Data')
plt.plot(x_values, y_values, 'g-', label='Polynomial')

# Adding labels and legend
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('Degree 8 Interpolation: Steam Temperature vs Steam Pressure')
plt.legend()

plt.savefig('SteamInterpolationDeg8.png',dpi =200) #uncomment to save graph
# Showing the plot
plt.show()
plt.close()









"""
Now for Part 4
"""

Xi = [1,2,3,4,5]
Yi = [1,(1/2),(1/3),(1/4),(1/5)]

A4 = makepoly(Xi, Yi, 4)

x_values = np.linspace((1/4), 7, 50)
y_values = evalpoly(x_values, Xi, A4, 4)
y_actual = 1 / x_values

#plotting the polynomial and f(x) = 1/x
plt.plot(x_values, y_values, 'g-', label = 'Interpolated Polynomial')
plt.plot(x_values, y_actual, 'r-', label = 'f(x) = 1/x')

#Adding labels and axis
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Interpolation of f(x) = 1/x')
plt.legend()

plt.savefig('x^-1.png',dpi =200) #uncomment to save graph
#Showing the plot
plt.show()
plt.close()






"""
Part 5
Polynomial interpolation to:  f(x) = (1+25x^2)^{-1}
"""
x_list = [-1, -.875, -.75, -.625, -.5, -.375, -.25, -.125, 0, .125, .25, .375, .5, .675, .75, .875, 1]
y_list = [.038, .05, .066, .093, .138, .221, .39, .719, 1, .719, .39, .221, .138, .093, .066, .05, .038]

"""
#Degree 4 polynomial
x_1 = [x_list[0], x_list[4], x_list[8], x_list[12], x_list[len(x_list)-1]]
y_1 = [y_list[0], y_list[4], y_list[8], y_list[12], y_list[len(y_list)-1]]

A5 = makepoly(x_1, y_1, 4)

A5_x_values = np.linspace(-1, 1)
A5_y_values = evalpoly(A5_x_values, x_1, A5, 4)


#Degree 8 polynomial
x_2 =[x_list[0], x_list[2], x_list[4], x_list[6], x_list[8], x_list[10], x_list[12], x_list[14], x_list[16]]
y_2 =[y_list[0], y_list[2], y_list[4], y_list[6], y_list[8], y_list[10], y_list[12], y_list[14], y_list[16]]

B5 = makepoly(x_2, y_2, 8)

B5_x_values = np.linspace(-1., 1.)
B5_y_values = evalpoly(B5_x_values, x_2, B5, 8)


#Degree 16 polynomial
x_3 = x_list
y_3 = y_list

C5 = makepoly(x_3, y_3, 16)

C5_x_values = np.linspace(-1, 1)
C5_y_values = evalpoly(C5_x_values, x_3, C5, 16)
"""

#Actual function of:   f(x) = (1+25x^2)^{-1}


def runge(x):
    return (1+25*(x**2))**(-1)


n4 = 4
x_1 = np.linspace(-1, 1, n4+1)
y_1 = runge(x_1)


A5 = makepoly(x_1, y_1, n4)

A5_x_values = np.linspace(-1, 1, 100)
A5_y_values = evalpoly(A5_x_values, x_1, A5, n4)



#Degree 8 polynomial
n8 = 8
x_2 = np.linspace(-1, 1, n8+1)
y_2 = runge(x_2)


B5 = makepoly(x_2, y_2, n8)

B5_x_values = np.linspace(-1, 1, 100)
B5_y_values = evalpoly(B5_x_values, x_2, B5, n8)



#Degree 16 polynomial
n16 = 16
x_3 = np.linspace(-1, 1, 100)
y_3 = runge(x_3)


C5 = makepoly(x_3, y_3, n16)

C5_x_values = np.linspace(-1, 1, 100)
C5_y_values = evalpoly(C5_x_values, x_3, C5, n16)



#Actual points
x_val_act = np.linspace(-1, 1, 100)
y_val_act = runge(x_val_act)





#plotting Degree 4 Interpolation
plt.plot(A5_x_values, A5_y_values, 'b-', label = 'Degree 4 Interpolation')
plt.plot(x_val_act, y_val_act, 'purple', label = 'Actual function points')

#Adding labels and axis
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Degree 4 interpolation of f(x) = (1+25x^2)^(-1)')
plt.legend()
plt.savefig('RungeDeg4.png',dpi =200)
#Showing the plot
plt.show()
plt.close()



#Plotting Degree 8 interpolation
plt.plot(B5_x_values, B5_y_values, 'r-', label = 'Degree 8 Interpolation')
plt.plot(x_val_act, y_val_act, 'purple', label = 'Actual function points')

#Adding labels and axis
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Degree 8 interpolation of f(x) = (1+25x^2)^(-1)')
plt.legend()
plt.savefig('RungeDeg8.png',dpi =200)
#Showing the plot
plt.show()
plt.close()



#Plotting Degree 16 Interpolation
plt.plot(C5_x_values, C5_y_values, 'g-', label = 'Degree 16 Interpolation')
plt.plot(x_val_act, y_val_act, 'purple', label = 'Actual function points')

#Adding labels and axis
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Degree 16 interpolation of f(x) = (1+25x^2)^(-1)')
plt.legend()
plt.savefig('RungeDeg16.png',dpi =200) 
#Showing the plot
plt.show()
plt.close()


# Create a figure with three subplots side by side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plotting the first graph on the first subplot
axs[0].plot(A5_x_values, A5_y_values, 'b-', label='Degree 4 Interpolation')
axs[0].plot(x_val_act, y_val_act, 'purple', label='Actual function points')
axs[0].set_xlabel('x values')
axs[0].set_ylabel('y values')
axs[0].set_title('Degree 4 interpolation of f(x) = (1+25x^2)^(-1)')
axs[0].legend()

# Plotting the second graph on the second subplot
axs[1].plot(B5_x_values, B5_y_values, 'r-', label='Degree 8 Interpolation')
axs[1].plot(x_val_act, y_val_act, 'purple', label='Actual function points')
axs[1].set_xlabel('x values')
axs[1].set_ylabel('y values')
axs[1].set_title('Degree 8 interpolation of f(x) = (1+25x^2)^(-1)')
axs[1].legend()

# Plotting the third graph on the third subplot
axs[2].plot(C5_x_values, C5_y_values, 'g-', label='Degree 16 Interpolation')
axs[2].plot(x_val_act, y_val_act, 'purple', label='Actual function points')
axs[2].set_xlabel('x values')
axs[2].set_ylabel('y values')
axs[2].set_title('Degree 16 interpolation of f(x) = (1+25x^2)^(-1)')
axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.savefig('RungeAll.png',dpi =200)
# Show the plot
plt.show()
plt.close()






if __name__ == '__main__':

    '''
    f(x) = x^2 -4, but done using Horner's rule
    '''
    def f(x):
        a = np.array([-4,0,1])
        n = len(a)
        y = a[n-1]
        for i in range(n-1):
            y = y*x + a[n-2-i]
        return y

  
    xn = np.array([-2,0,2])
    yn = f(xn)
    N = 20
    try:
        a = makepoly(xn,yn,N)
    except ValueError as e:
        msg = f"Requested a polynomial of degree {N} but only provided {len(xn)} nodes."
        assert str(e) ==msg
    xn2 = np.linspace(-2,0, N+1)
    yn2 = f(xn2)
    a = makepoly(xn2,yn2,N)
    try:
        yn3 = evalpoly(xn2,xn2,a[0:3],N)
    except ValueError as e:
        msg = f"Requested a polynomial of degree {N} but only provided 3 coefficients."
        assert str(e)==msg
    try:
        yn3 = evalpoly(xn2,xn2[0:3],a,N)
    except ValueError as e:
        msg = f"Requested a polynomial of degree {N} but only provided 3 nodes."
        assert str(e) == msg
    N = 2
    a = makepoly(xn,yn,N)
    x = np.linspace(-5,5,50)
    y = evalpoly(x,xn,a,N)
    y1 = f(x)
    eps = 1e-13
    assert(all(abs(y-y1)/abs(y1)<eps))

    def g(x):
        return -3*x**3 + 2*x**2 -1*x + 4

    xn = np.array([-3, -2, -1, 0])
    yn = g(xn)
    N = 3
    a = makepoly(xn,yn,N)
    x = np.linspace(-np.pi, np.pi, 30)
    y =evalpoly(x,xn,a,N)
    y1 = g(x)
    assert(all(abs(y-y1)<eps))

    '''
    ================================================================
    Example from class  done by hand
    this calculation should be exact or close to it, so I used a very small
    epsilon to test with.
    ================================================================
    '''
    eps = 1e-15 
    xn = np.array([1,2,3,4])
    yn = np.array([3,2,3,2])
    ahand = np.array([3,-1,1, -2/3])
    a= makepoly(xn,yn,3)
    '''
    check hand calculations vs. makepoly
    '''
    assert(all(abs(a-ahand)<eps))
    '''
    check that nodes are interpolated correctly
    '''
    y2 = evalpoly(xn,xn,a,3)
    assert(all(abs(yn-y2)< eps))
    
    ''' 
    Add tests and other work here
    '''
    