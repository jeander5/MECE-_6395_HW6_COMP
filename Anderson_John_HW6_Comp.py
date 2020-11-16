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
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import time
from math import sin as sin
from math import cos as cos


#Constants given in problem statement, constant for both boundary conditions
L=math.pi
D=0.1
T=10
#k is just an integer
k=1

#discretize the interval function, also works for time     
def DIF(L ,N):
    """discretizes an interval with N interior points"""
#Inputs are interval lengths number of interior points    
    h = L/(N+1)
    x = np.linspace(0, L, N+2)
    return(x, h)
    
def thomas_alg_func(a,b,c,f):
    """solves tridiagonal matrix"""
#inputs are vectors containing the tridiagonal elements and right hand side
    N=len(a)
#storage vectors    
    u_appx = [0]*N
    alpha = [0]*N
    g = [0]*N
#Following the pseudocode
#Zeroth element of this list corresponds to the first subscript in Thomas Algorithm
    alpha[0] = a[0]
    g[0] = f[0]
    for j in range(1, N):
        alpha[j] = a[j]-(b[j]/alpha[j-1])*c[j-1]
        g[j] = f[j]-(b[j]/alpha[j-1])*g[j-1]
    u_appx[N-1] = g[N-1]/alpha[N-1]
    for j in range(1, N):
        u_appx[-1-j] = (g[-1-j]-c[-1-j]*u_appx[-j])/alpha[-1-j]
    return u_appx    

#u exact function part a
def u_exact_func_a(k, D, t, x):
    """returns exact u values for the function from Part a"""
#Inputs are x and t values, and the given constants
    # i geuss I need to use a for loop here. I would like to do some like this:
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

#dont really think I need functions for these but I made the.
#it is nice to have I can change the inputs on the fly    
def BC_Partb(t,w,k,L):
    """returns boundary conditions for the function from Part b"""
#inputs are the time list and the given constants
    a=cos(k*L)
    g_0b =[sin(w*t) for t in t]
    g_Lb= [sin(w*t)*a for t in t]
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

def avg_error (exact,appx):
    """returns average error"""
#inputs are just single row for exact and approximate solution
#the entire x domain at a single time    
    N=len(exact)
    mysum =0
    for j in range(1,N-1):
        mysum=mysum+abs((appx[j]-exact[j])/exact[j])
    error=mysum/N
    return error


# N right here for now, just using the same for x and T
N = 400

#calling the DIF
x, dx = DIF(L,N)
t, dt = DIF (T,N)
#lets define omega right here actually, after dt is defined
w=0.1/dt 
len_x = len(x)
len_t = len(t)

#and I will define the CN constants right here as well since I have dx and dt

#constants from the CN scheme used in the thomas algorithm
#i know how to spell lambda, put python has lambda functions so I write lamda
lamda=D*dt/(dx*dx)
b=-lamda/2
a=1+lamda
c=-lamda/2
#im defining a d so it doesnt have to calculate inside loops or functions
d=1-lamda

#Part a
#values given in problem statement
g_0=0
g_L=0
f=[sin(k*x) for x in x]
F= np.zeros(shape=(len_t, len_x))

#solution matrix, will be filled in
u_appx_a=np.zeros(shape=(len_t, len_x))

#Boundary Conditions
u_appx_a[:,0]=g_0
u_appx_a[:,-1]=g_L
#Initial Conditions
u_appx_a[0,1:-1]=f[1:-1]
#I still have [1:-1] no need to reassign those endpoints for t =0

#input vecotrs for the thomas algorithm function
av= [a]*N
bv= [b]*N
cv= [c]*N
rhs= [0]*N

#for loop for calling the thomas algorithm at the different time steps
for n in range (1,len_t):
    Q=F[n-1,:]
    q=u_appx_a[n-1,:]
# first input different    
    rhs[0]=-b*q[0]+d*q[1]-c*q[2]-b*u_appx_a[n,0]+dt*Q[1]
#    inner for loop for filling up the rhs vector for the thomas algorithm    
    for j in range (1,N-1):
        rhs[j]=-b*q[j]+d*q[j+1]-c*q[j+2]+dt*Q[j+1] 
    #last input different
        j=j+1
        rhs[j]=-b*q[j]+d*q[j+1]-c*q[j+2]-c*u_appx_a[n,-1]+dt*Q[j+1]      
    u_appx_a[n,1:-1]=thomas_alg_func(av,bv,cv,rhs)


#calling exact function
u_exact_a=u_exact_func_a(k, D, t, x)
#calling the error function
error_a=avg_error(u_exact_a[-1,:],u_appx_a[-1,:])


# =============================================================================
# part B
# =============================================================================
#is not working

#values given in problem statement
g_0,g_L=BC_Partb(t,w,k,L)
#initial condition
f=[0]*len_x
F=preF(D,k,w,x,t)

#solution matrix, will be filled in
u_appx_b=np.zeros(shape=(len_t, len_x))

#Boundary Conditions
u_appx_b[:,0]=g_0
u_appx_b[:,-1]=g_L
#Initial Conditions
u_appx_b[0,:]=f

#input vecotrs for the thomas algorithm function
#i dont really need to redine these but im trying to be consistent
av= [a]*N
bv= [b]*N
cv= [c]*N
rhs= [0]*N

#for loop for calling the thomas algorithm at the different time steps
for n in range (1,len_t):
#from 1, len_t because initial conditions are defined
    Q=F[n-1,:]
    q=u_appx_b[n-1,:]
#also look at this line    
#    rhs[0]=-b*q[0]+d*q[1]-c*q[2]-b*u_appx_b[n,0]+dt*Q[1]
#check the signs    
    rhs[0]=-b*q[0]+d*q[1]-c*q[2]-b*u_appx_b[n,0]+dt*Q[1]
#    inner for loop for filling up the rhs vector for the thomas algorithm
#    print(n)
    for j in range (1,N-1):
#this is the line I need to focus on
#pay attention to indexes, but I dont think that is the issue        
#        rhs[j]=-b*q[j]+d*q[j+1]-c*q[j+2]+dt*Q[j+1]        
        rhs[j]=-b*q[j]+d*q[j+1]-c*q[j+2]+dt*Q[j+1]        
#last input different
        j=j+1
        rhs[j]=-b*q[j]+d*q[j+1]-c*q[j+2]-c*u_appx_b[n,-1]+dt*Q[j+1]  
                  
        
#        print(j)
    u_appx_b[n,1:-1]=thomas_alg_func(av,bv,cv,rhs)

#calling exact function
u_exact_b=u_exact_func_b(k, D, w, t, x)
#calling the error function
error_b=avg_error(u_exact_b[-1,:],u_appx_b[-1,:])

#I still have more error for part b tho
#next up is plotting and grid convergence

#plt.plot(x,u_appx_b[-1])

#Do I wanna do this in a while loop like last time?
# or just do it manually calling the functions to get the graphs I need?

#I kinda wanna make a sweet graphing function. Like input (N_x, N_t, a, 8)
#and it will plot 8 differnt graphs with the exact function for N_x points, 2*N_x points ect ect
  