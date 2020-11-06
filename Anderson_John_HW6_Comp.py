# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:11:56 2020

@author: johna
"""

# MECE 6397, SciComp, HW 6, Computational
# 1-D Diffusion Problem Crank Nicolson
#https://github.com/jeander5/MECE_6397_HW6_COMP

#imports
import math
import numpy as np
from math import sin as sin

#import matplotlib.pyplot as plt

#Constants given in problem statement, constant for both boundary conditions
#using case a) for now

L=math.pi
D=0.1
T=10
F=0
g_0=0
g_L=0
#k is just an integer
k=1

#functions

#discretize the interval function, for time and length     
def DIF(L,N):
#Discretizing the interval length. This is the same for both problems
    h = L/(N+1)
    x = np.linspace(0, L, N+2)
    return(x[:],h)

# Thomas Algorithm Function, for part a Crank Nicholson Scheme part a
#lets try to be smarter about this compared to last time, add only what I need in the function

def thomas_alg_a(N,a,b,c,d,f):
    """returns approximnte u values for the function from Part 1"""
#inputs are N interavl length, the tridiagonal elements, which are scalars here
#and f, which is the just the right hand side, which is a vector  
    
#Pre Thomas Algorithm set up. For this problem these values are all constant
    alpha = [0]*N
    g = [0]*N
    u_appx = [0]*N
#Following the pseudocode
#Zeroth element of this list corresponds to the first subscript in Thomas Algorithm
    alpha[0] = a
    g[0] = -b*f[0]+d*f[1]-c*f[2]
    for j in range(1, N):
        rhs=-b*f[j]+d*f[j+1]-c*f[j+2]
        alpha[j] = a-(b/alpha[j-1])*c
        g[j] = rhs-(b/alpha[j-1])*g[j-1]
    u_appx[N-1] = g[N-1]/alpha[N-1]
    for j in range(1, N):
        u_appx[-1-j] = (g[-1-j]-c*u_appx[-j])/alpha[-1-j]
    return u_appx

#u exact function
def u_exact_funca(k, D, t, x):
    """returns exact u values for the function from Part 2"""
    # i geuss I need to use a for loop here I would like to do some like this:
#    f= sin(x*t) for x in x and t and t
    len_x = len(x)
    len_t = len(t)
    func_vals=np.zeros(shape=(len_t, len_x))
    for n in range(len_t):
        func_vals[n,1:-1]=[math.exp(-D*k*k*t[n])*math.sin(k*x) for x in x[1:-1]]
#note I have from x in x[1:-1] in here to just keep the values as zero so I dont need to reassign them 
# the boundary conditions u(x=0,t) and  u(x=L,t) are zero for all t         
    return func_vals



# N right here for now, just using the same for x and T
N = 25

#calling the DIF

x, dx = DIF(L,N)
t, dt = DIF (T,N) 
len_x = len(x)
len_t = len(t)

#might as well call u_exact rn
u_exact=u_exact_funca(k, D, t, x)

#okay now lets apply the scheme for part a

#zeros u_appxution matrix
#we are gonna fill this baby up as we go
u_appx=np.zeros(shape=(len_t, len_x))
#rows are time steps, columns x position
#filling in initial and boundary conditions
# the boundary conditions u(x=0,t) and  u(x=L,t) are zero for all t  so I just leave as zeros
# this will need to be modified for part b
# nah lets just right it generally right now

#Boundary Conditions
u_appx[:,0]=g_0
u_appx[:,-1]=g_L
#Initial Conditions
f=[sin(k*x) for x in x]
u_appx[0,1:-1]=f[1:-1]
#I still have [1:-1] no need to reassign those endpoints for t =0

#on to the scheme......
#i know how to spell lambda, put python has lambda functions so I write lamda
lamda=D*dt/(dx*dx)
b=-lamda/2
a=1+lamda
c=-lamda/2
#im defining a d so it doesnt have to calculate inside loops or functions
d=1-lamda

#im not really seeing a better way to do this besides calling that function inside a for loop and
#filling up my u_appx matrix
for n in range (1,len_t):
    #I wonder if its inefficient to have u_appx[n-1,: in the function call, I bet it is.
    #I will just define a new variable    
    q=u_appx[n-1,:]
    u_appx[n,1:-1]=thomas_alg_a(N,a,b,c,d,q)
#looks good to me    
#lets make a git hub    