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
from math import cos as cos

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
def DIF(L ,N):
#Discretizing the interval length. This is the same for both problems
    h = L/(N+1)
    x = np.linspace(0, L, N+2)
    return(x, h)

# Thomas Algorithm Function, for part a Crank Nicholson Scheme part a
#lets try to be smarter about this compared to last time, add only what I need in the function

def thomas_alg_a(N,a,b,c,d,f):
    """returns approximate u values for the function from Part 1"""
#inputs are N interavl length, the tridiagonal elements, which are scalars here
#and f, which is the just the right hand side, or the u_n elements from Crank Nicolson
    
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

def thomas_alg_b(N,a,b,c,d,f,F,g0_b):
    """returns approximate u values for the function from Part 1"""
#inputs are N interavl length, the tridiagonal elements, which are scalars here
#and f, which is the just the right hand side, or the u_n elements from Crank Nicolson
#also in this function we have the prescribed function F which contributes to the rhs
#and the boundary condition g0_b (u x=0) which contributes to the first rhs equation    
    
    alpha = [0]*N
    g = [0]*N
    u_appx = [0]*N
#Following the pseudocode
#Zeroth element of this list corresponds to the first subscript in Thomas Algorithm
    alpha[0] = a
    g[0] = -b*f[0]+d*f[1]-c*f[2]+F[1]-g0_b*b
    for j in range(1, N):
        rhs=-b*f[j]+d*f[j+1]-c*f[j+2]+F[1+j]
        alpha[j] = a-(b/alpha[j-1])*c
        g[j] = rhs-(b/alpha[j-1])*g[j-1]
    u_appx[N-1] = g[N-1]/alpha[N-1]
    for j in range(1, N):
        u_appx[-1-j] = (g[-1-j]-c*u_appx[-j])/alpha[-1-j]
    return u_appx

#I will maybe make these 1 function with the same inputs later.
#right now they are seperate because of the different boundary conditions
#and the different values for prescribed function F

#u exact function part a
def u_exact_func_a(k, D, t, x):
    """returns exact u values for the function from Part a"""
#Inputs are x and t values, and the given constants
    # i geuss I need to use a for loop here I would like to do some like this:
#    f= sin(x*t) for x in x and t and t
    len_x = len(x)
    len_t = len(t)
    func_vals=np.zeros(shape=(len_t, len_x))
    for n in range(len_t):
        func_vals[n,1:-1]=[math.exp(-D*k*k*t[n])*sin(k*x) for x in x[1:-1]]
#note I have from x in x[1:-1] in here to just keep the values as zero so I dont need to reassign them 
# the boundary conditions u(x=0,t) and  u(x=L,t) are zero for all t         
    return func_vals

#u exact function part b
def u_exact_func_b(k, D, w, t, x):
    """returns exact u values for the function from Part b"""
#Inputs are x and t values, and the given constants    
    len_x = len(x)
    len_t = len(t)
    func_vals=np.zeros(shape=(len_t, len_x))
    for n in range(len_t):
        func_vals[n,:]=[sin(w*t[n])*cos(k*x) for x in x]
    return func_vals

def BC_Partb(x,t,w):
    """returns boundary conditions for the function from Part b"""
#inputs are x and t list, and then weq which is the vlaue that w*dt is equal to
#or I can just input w because I need it later for exact value    
#    dt = t[-1]-t[-2]
#    w = weq/dt
    g_0b =[sin(w*t) for t in t]
    g_Lb= [sin(w*t)*cos(k*L) for t in t]
    return(g_0b, g_Lb)
    
#The prescribed function F
def preF(D,k,w,x,t): 
    """returns values for the prescribed function F(x,t)"""
#Inputs are x and t values, and the given constants    
    len_x = len(x)
    len_t = len(t)
    func_vals=np.zeros(shape=(len_t, len_x))    
    for n in range(len_t):
        func_vals[n,:]=[cos(k*x)*((w*cos(w*t[n])+D*k*k*sin(w*t[n]))) for x in x]
    return func_vals



# N right here for now, just using the same for x and T
N = 5

#calling the DIF
x, dx = DIF(L,N)
t, dt = DIF (T,N)
#lets define omega right here actually, after dt is defined
w=0.1/dt 
len_x = len(x)
len_t = len(t)

#part a

#zeros u_approximation matrix
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

#constants from the CN scheme used in the thomas algorithm
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
   
#calling exact function
u_exact=u_exact_func_a(k, D, t, x)


#part b

#set up solution matrix    
u_appx_b=np.zeros(shape=(len_t, len_x))
#fill in boundary conditions, need to call that function.
g0_b,gL_b=BC_Partb(x,t,w)
u_appx_b[:,0]=g0_b
u_appx_b[:,-1]=gL_b

#The Initial conditions are just zero so no need to reassign them
#calling the prescribed function F
F=preF(D,k,w,x,t)
# the constants a,b,c,d are the same for part 1

#filling up my matrix
for n in range (1,len_t):
    q=u_appx_b[n-1,:]
    #that prescribed function F needs to be multiplied by dt
    Q=F[n-1,:]*dt
    u_appx_b[n,1:-1]=thomas_alg_b(N,a,b,c,d,q,Q,u_appx_b[n,0])

#calling exact function
u_exact_b=u_exact_func_b(k, D, w, t, x)    

#next I need to do error and grid convergence. For the grid convergence I dont think I will do it 
#in a while loop like for HW 4, I will just do it manually and save the graphs, tables ect.
#its already pretty close with not that many grid points