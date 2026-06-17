#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 6 2025
"""

import numpy as np
import networkx as nx
import scipy.optimize
import scipy.integrate as defint
import matplotlib.pyplot as plt 
from numba import njit, prange, types
from numba.typed import List

####### Constructing pdfu ###################3
# Finding total volume under the pdf
def vol_in_u(h0, omega, tm):
    def dV(u):     #functional form of sqrt(-g)
        return 1 - 0.5*(h0**2)*(np.cos(omega*u)**2)
    volin_u = defint.quad(dV,-tm/2,tm/2)  #ranges are in the order u,v,x,y as the order in the def of dV
                                        # I think the volume factor from integration over v,x,y gets cancelled out theoretically
                                        # b/c dV/volin_u[0] is actually (
                                        # volin_remaindom=integral over \delta v,x,y)/volin_remaindom * dV/volin_u[0]
    return volin_u[0]                   #quad returns an array! answer, error. vol[0] chooses the answer

# Define pdfu(u) 
@njit                           
def pdfu(arg, h0, omega, tm, volin_u):
    #the pdfu we are using for an interval of sqrt(2)*10
    #for symbolic integration https://www.freecodecamp.org/news/calculate-definite-indefinite-integrals-in-python/
    #for converting symbolic ouput to a function https://stackoverflow.com/questions/40609184/python-changing-symbol-variable-and-assign-numerical-value
    ##continued https://www.tutorialspoint.com/sympy/sympy_lambdify_function.htm
   
    def dV(u):     #functional form of sqrt(-g)
        return 1 - 0.5*(h0**2)*(np.cos(omega*u)**2) 
    output = dV(arg)/volin_u 
    return output# no need to integrate dV!!

# Find maximum value for pdfu
def find_c(h0, omega, tm, volin_u):
    umax = scipy.optimize.fmin(lambda u: -pdfu(u, h0, omega, tm, volin_u),0) # finds the first u where function is max by finding the minimum of -pdfu 
                                                     # (which is the max of +pdfu)
                                                     # need lambda because fmin doesn't accept -pdfu directly
    c = pdfu(umax, h0, omega, tm, volin_u)
    return c

################## Generating coordinates ###################3
# njit wrapped function to generate the coordinates
@njit
def GW_fc(N, D, h0, omega, tm, c, volin_u):
    coords = np.zeros((N,D))
    temp_coord = np.zeros((1,D))
    successes=0
    
    while successes < N:
        #using rejection sampling to generate coordinates of u = t-z
        #comparison distribution is a uniform one between -tm/2,+tm/2, call it U_1, of height c=max(pdfu)
        ########## r = (b-a)*rand1 + a = timeinterval*sqrt(2)*rand1 - tm/sqrt()
        #acceptance criterion if U_2, which is another uniform distribution betweeen 0 and c, is < pdfu(x)
        U_1=c
        r = tm*np.random.random() - tm*0.5 ######## u ranges from -timeinterval*sqrt(2)/2,+timeinterval*sqrt(2)/2.
        U_2 = U_1*np.random.random()
        
        if U_2 > pdfu(r, h0, omega, tm, volin_u):
            continue # while loop starts again and x is automatically rreplaced by a new temp
            
        temp_coord[0,0] = r # new r accepted
        
        #generate v in range 0, tm
        temp_coord[0,1] = tm*np.random.random()
        
        #generate x and y in range -timeinterval/2,timeinterval/2 where timeinterval=tm/sqrt(2)
        temp_coord[0,2:] = tm/np.sqrt(2) * np.random.rand(1,D-2) -  tm/np.sqrt(2)*0.5
        
        coords[successes,:] = temp_coord[0,:]
        
        successes += 1
    # Assign a natural ordering. t = u + v (/2 as well, but that doesn't change the ordering). 
    # The code creates a temp array of the sum of the u and v columns and sorts it
    # and spits out the sorted indices, which then go back into coords to sort coords
    final_coords = coords[np.argsort(coords[:,0] + coords[:,1])]
    
    return final_coords

#################### Generating packed R ########################
@njit
def make_empty_packed(N):
    # Ceil division for bytes-per-row
    W = (N + 7) // 8
    return np.zeros((N, W), dtype=np.uint8)

@njit
def set_edge(Rp, i, j):
    byte_idx = j >> 3           # same as j // 8
    bit_pos  = 7 - (j & 7)      # MSB-first within each byte
    Rp[i, byte_idx] |= (1 << bit_pos)

@njit
def clear_edge(Rp, i, j):
    byte_idx = j >> 3
    bit_pos  = 7 - (j & 7)
    Rp[i, byte_idx] &= ~(1 << bit_pos)

@njit
def get_row_neighbors(Rp, u, N):
    """
    Unpack row u of Rp and return a typed.List of all j where (u→j) is present.
    Only O(degree(u)) extra memory is used for the list.
    """
    W = (N + 7) // 8
    nbrs = List.empty_list(types.int64)   # start empty

    for bidx in range(W):
        byte = Rp[u, bidx]
        if byte:
            # scan the 8 bits
            for bit in range(8):
                if (byte >> (7 - bit)) & 1:
                    j = bidx*8 + bit
                    if j < N:
                        nbrs.append(j)
    return nbrs

# Set number of threads to be used by numba. Might be important for running on cluster
# set_num_threads(8)

@njit(parallel=True)
def R_GW(fc, omega, h0):
    N = len(fc)
    
    # Initialize R
    R_packed = make_empty_packed(N)
    
    E_count = 0
    for i in prange(0,N-1):
        for j in range(i+1,N-1):
            du=fc[j,0]-fc[i,0];dv=fc[j,1]-fc[i,1];dx=fc[j,2]-fc[i,2];dy=fc[j,3]-fc[i,3];u1=fc[i,0];u2=fc[j,0] #;v1=fc[i,1];v2=fc[j,1]
            
            norm_W0 = -2*du*dv + dx**2 + dy**2 + 2*h0/(omega*du)*(dx**2-dy**2)*np.sin(omega*u2/2)*np.cos(omega*(u1+u2)/2)
            
            if norm_W0 < 0:
                set_edge(R_packed, i, j)
                E_count += 1
    return R_packed, E_count

######### Wrapper function using all the functions together 
from time import perf_counter

def plusGW(numofpoints = 1000, numofwaves=1, timeinterval=10):
    #Finding a numerical value for V for h=0.1, timeinterval=10, \\[Omega]=2\[Pi](1/\[Lambda]) where \[Lambda] is the wavelength. 
    #We want there to be two waves in a region of 10x10x10 so \\[Lambda]=tm/2=5 so \[Omega]=2\[Pi]/5*)
    D=4
    N = numofpoints
    lamb = timeinterval/numofwaves
    omega = 2*np.pi/lamb
    h0=1/(2*np.pi**2*numofwaves**2)##Using h_0^max of the prospectus eq.2.24
    # print(h0)
    # timeinterval = 10 # fixing the timeinterval becauze this is the interval for which I have the exact form of the pdfu in mathematica.
                        # If I want a general time interval then I would have to find the pdfu in this code insterad of mathematica.
    
    tm=np.sqrt(2)*timeinterval #changing the nomenclature used in onenote notes. v range earlier was tm*sqrt(2) now I am calling this tm. 
                                #and tm in notes as timeinterval in the code.
    t0 = perf_counter()
    # Total volume of pdf
    volin_u = vol_in_u(h0, omega, tm)
    t1 = perf_counter()
    print(f'Time to find volin_u: {t0 - t1:.2f}s')
    
    # Max height of pdf
    c = find_c(h0, omega, tm, volin_u)
    t2 = perf_counter()
    print(f'Time to find c: {t2 - t1:.2f}s')
    
    ###################### Generate fc ############
    fc = GW_fc(N, D, h0, omega, tm, c, volin_u)
    t3 = perf_counter()
    print(f'Time to make fc: {t3 - t2:.2f}s')
    
    ###################### Make R_packed ##########
    R_packed, E_count = R_GW(fc, omega, h0)
    t4 = perf_counter()
    print(f'Time to make R_packed: {t4 - t3:.2f}s')
    print(f'E_count for N = {N/1000:.0f}k is: {E_count:.2e}')
    return fc, R_packed
    
if __name__ == "__main__":
    import tracemalloc # To track aximum memory requested by python

    tracemalloc.start()

    fc, R_packed = plusGW(200000, 1, 21)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory allocation peak: {peak/1024**2:.2f} MB")
    tracemalloc.stop()