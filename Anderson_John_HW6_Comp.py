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
#try different increasing values of w, from wdt = 0.1 on up.
w=0.25
#anonoymous lambda functions for boundary conditions part b
#hmmm I cant make the inputs to these a list...lets just make them a regular function
#g_0b=lambda t: sin(w*t)
#g_Lb=lambda t,x: sin(w*t)*cos(k*L)

def BC_Partb(x,t,w):
    """returns boundary conditions for the function from Part b"""
#inputs are x and t list, and then weq which is the vlaue that w*dt is equal to
#or I can just input w because I need it later for exact value    
#    dt = t[-1]-t[-2]
#    w = weq/dt
    g_0b =[sin(w*t) for t in t]
    g_Lb= [g_0b*cos(k*L) for g_0b in g_0b]
    return(g_0b, g_Lb)
    
    
#I also have that Capitol F function, what should I call that, its not really a Forcing Functions
#The prescribed function
def preF(D,k,w,x,t): 
    """returns values for the prescribed function F(x,t)"""
    len_x = len(x)
    len_t = len(t)
    func_vals=np.zeros(shape=(len_t, len_x))    
    for n in range(len_t):
        func_vals[n,:]=[cos(k*x)*((w*cos(w*t[n])+D*k*k*sin(w*t[n]))) for x in x]
    return func_vals
#Inputs are x and t values, and the given constants    

        
        
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



# N right here for now, just using the same for x and T
N = 8
#calling the DIF

x, dx = DIF(L,N)
t, dt = DIF (T,N*2)
#lets define omega right here
w=0.1/dt 
len_x = len(x)
len_t = len(t)

#might as well call u_exact rn
u_exact=u_exact_func_a(k, D, t, x)

#okay now lets apply the scheme for part a

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
    
#okay lets move on to part b
#ok got my functions
#now lets do my algorithm
# I know have F which is part of the right hand side of my algorithm for all values of T
# should I modify the input? create a new function? modify the function? 
#I think I can just modify the input    
#oh but f(x,t=0) is just zero for this case so it is actually kinda easier
#I will just quickly create a new function and try and combine them later

def thomas_alg_b(N,a,b,c,d,f,F):
    """returns approximate u values for the function from Part 1"""
#inputs are N interavl length, the tridiagonal elements, which are scalars here
#and f, which is the just the right hand side, which is a vector  
    
#Pre Thomas Algorithm set up. For this problem these values are all constant
    alpha = [0]*N
    g = [0]*N
    u_appx = [0]*N
#Following the pseudocode
#Zeroth element of this list corresponds to the first subscript in Thomas Algorithm
    alpha[0] = a
    g[0] = f[0]+F[1]
    for j in range(1, N):
        rhs=-b*f[j]+d*f[j+1]-c*f[j+2]+F[1+j]
        alpha[j] = a-(b/alpha[j-1])*c
        g[j] = rhs-(b/alpha[j-1])*g[j-1]
    u_appx[N-1] = g[N-1]/alpha[N-1]
    for j in range(1, N):
        u_appx[-1-j] = (g[-1-j]-c*u_appx[-j])/alpha[-1-j]
    return u_appx

#okay set up solution matrix
    
SOL=np.zeros(shape=(len_t, len_x))

#fill in boundary conditions, initial conditions are just zero
g0_b,gL_b=BC_Partb(x,t,w)
SOL[:,0]=g0_b
SOL[:,-1]=gL_b
#calling the prescribed function F
F=preF(D,k,w,x,t)
# the constants a,b,c,d are the same for part 1

#filling up my SOL matrix
for n in range (1,len_t):
    #I will just define a new variable so I am not returning the f initial condtion function     
    q=SOL[n-1,:]
    #that prescribed function F needs to be multiplied by dt
    Q=dt*F[n-1,:]
    SOL [n,1:-1]=thomas_alg_b(N,a,b,c,d,q,Q)

#calling exact function

UB=u_exact_func_b(k, D, w, t, x)    

#maybe I should change all these list to arrays


#i have an error somewhere its in the algorith or calling the algorithm
# i think it has to do with the F
#nope! it was a copy and pasting issue! Okay good. 
#i had SOL[0] as u_appx[0] from the part a when calling
#its still not perfect though, there is another error

#and Im not seeing what ele to do. Im adding F[0,1], which corresponds to rhs or the f in the algorith
#and Im multiplying F by dt*
# Im just not seeing it

#when I do the scheme I am first at n=0 and solving for n=0+dt*1
#So I have been doing F at t=0 into the algorithm 
#but maybe it should be F at t=0+dt

#maybe I have a row or a column switched around somewhere I dont think so tho

#take a break, comeback later

#its gotta be in the algorithm

