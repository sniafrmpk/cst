#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:09:34 2025

@author: sheikhnaumanibrahimahmed
"""

from numba import njit
from multiprocessing import Pool
import tqdm
import time
from scipy.sparse import csr_matrix, save_npz
import os
from os.path import join, expanduser
import random
import scipy as sp
import numpy as np

############# Generate fc for a Minkowski cube ###############
def Minkowski_cube(rho = 10000, D = 4, l=0):
   
    ###define things that we need###
    #N is the total number of points
    N=rho
    height = 1.0
    #coords is the unsorted array of coordinates of the points in the hypercube
    coords = np.zeros((N,D)) # this creates an array with N rows and D columns
    
    # Force coding the two points of interest, a and b, at (1,-l/2,0,0) and (1,l/2,0,0)
    # This is not needed for finding relationship between the thickness of the tube and N
    coords[0,:]=(1,-l/2,0,0)
    coords[1,:]=(1,l/2,0,0)
    #and also the coordinates for c, the comparison point to be at (0,0,0,0):
    coords[2,:]=(0,0,0,0)
    #coords[N-1,0] = height
    #trial_coord are the coordinates of one point. It will be generated repeatedly
    temp_coord = np.zeros((1,D)) #defining and initializing the temp coord as an array of
    #a single row with D columns filled with zeros. two braces are necessary in using np.zeros. 
    #np.zeros((1)) the same as np.zeros((1,1)).
    successes = 3 #defining the counter that tells when the coords array is full
    
    while successes < N: #colon is need for loops apparently
     
    #generate temp_coord where the domain of the cube is [0,1] in time and [-0.5,0.5] for x, y, z.
        temp_coord[0,0] = height * random.random() #creates the time coordinate
        temp_coord[0,1:] = height * np.random.rand(1,D-1) - height * 0.5 # this simultaneously
    #creates all the space coordinates of the temp coord. the magic happens by
    #using "1:" in the arg of the temp_coord because it means all the columns 1 and 
    #onwards. Then np.random.rand(1,D-1) generates an array of random numbers between
    #0-1 with one row and D-1 columns. we are subtracting half the height because
    #we want our cube to be symmetric along the spatial axes (axes is the plural of axis).
    
    
        coords[successes,:] = temp_coord[0,:] #else if command to accept the temp coord
         #and add it to the coordinates array. no need to write coords(successes,0:) i.e.
         #a 0 before the :.......... Also I think temp_coord[:] without the 0, would have 
         #worked just fine.
    
        successes += 1
   
    # Inducing natural labeling on the points by sorting them according to the time coordinate of each point##
    #Where will points a,b and c end up at? I want to know their index. because all points have a definte time
    #coordinate and they are sorted based on that therefore all of them will have an index based on their time
    #coordinate. For c, this would mean that it will mean that it will get the index 0 because none of the randomly
    #generated points can have a time coordinate that is exactly zero. For a and b it will be tricky because I have
    #assigned both of them a time of coordinate value of 1. I guess the thing to do is to just run the sort and 
    #see what I get! OK, so consistently I see that b with x_b=+l/2 gets placed one index below a with x_a=-l/2.
    
    ###########So, that tells me that a has index N-1 and b has index N-2 and c has index 0.######################
    
    #sort_start=time.monotonic() # start time to check how long it takes for np.argsort to do the sorting
    final_coords = coords[np.argsort(coords[:,0])]
    return final_coords

############# Generate the arrays for R_sparse ###############
@njit
def R_sparse_arrays(final_coords):
    N = len(final_coords)
    
    # Estimate the maximum number of relations (upper bound for memory preallocation)
    max_relations = int(N * N/2)  # Adjust based on expected sparsity
    
    row_indices = np.empty(max_relations, dtype=np.int32)
    col_indices = np.empty(max_relations, dtype=np.int32)
      

    count = 0  # Counter for actual relations found
    
    for i in range(N): # this cannot be prange because the value of count is not updated fast enough
                        # neither do I want to do something like row_indices.append because I am afraid that 
                        # the corresponding col indiex will not be stored at the same time and the index of some
                        # other parallel execution will add to the column array.
        t_i, x_i, y_i, z_i = final_coords[i, 0], final_coords[i, 1], final_coords[i, 2], final_coords[i, 3]

        for j in range(i + 1, N):
            t_j, x_j, y_j, z_j = final_coords[j, 0], final_coords[j, 1], final_coords[j, 2], final_coords[j, 3]

            if (t_j - t_i)**2 > (x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2:
                row_indices[count] = i
                col_indices[count] = j
                count += 1 # gives the same count as for the the dense array method! 
                               # but the number of non-zero entries is less when csr constructed
                               # The col_indices can never contain 0s because j > 0 in all loops.
    
    values = np.ones(count, dtype=np.int8)   # All values will be 1                       
    numofrel = count
    
    return row_indices[:count], col_indices[:count], values[:count], numofrel # array[:n] = array[0:n], : is a slicing operator.

############# Create csr_matrix using the index and data arrays ############

def generate_save_R_fc(N):
    t0 = time.time()
    #initialize D and l for Minkowski cube
    D, l = 4, 0
    final_coords = Minkowski_cube(N, D, l)
    print(f"Generating the final coordinates took: {time.time()-t0:.2f} s")

    # Generate the arrays
    row_indices, col_indices, values, numofrel = R_sparse_arrays(final_coords)
    t1 = time.time()
    print(f"Finding the arrays for R_sparse took: {t1 - t0: .2f} s")
    
    # Make R_sparse from the arrays
    R_sparse = csr_matrix((values, (row_indices, col_indices)), shape=(N, N))
    t2 = time.time()
    print(f"Constructing the csr_matrix took: {t2 - t1: .2f} s")
    print(f"---Total time to construct R_sparse using arrays: {t2-t0:.2f} s")

    # Store fc and R on file
     # because the code will be run in parallel I want each pair of fc and R to have a unique timestamp from day, down to microseconds #####
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1_000) % 1_000}" # timestamp with milisecond precision. 
                                                                                        # 1_000 is the same as 1000. time.time() generates time in microseconds. % means mod.
    
    # defining the folder path
    parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Rs_fcs")
    folder_name = f'N_{int(N/1000)}k'
    folder_path = join(parent_path, folder_name)
    #creating folders
    os.makedirs(folder_path, exist_ok=True)
    
    # save R_sparse
    file_name_R = f'R_{int(N/1000)}k_{timestamp}.npz'
    file_path_R = join(folder_path, file_name_R)
    save_npz(file_path_R, R_sparse)
    t3 = time.time()
    print(f"---Time to save R as a npz file: {t3-t2:.2f} s")

    # save fc
    file_name_fc = f'fc_{int(N/1000)}k_{timestamp}.npy'
    file_path_fc = join(folder_path, file_name_fc)
    np.save(file_path_fc, final_coords)
    print(f"---Time to save fc as a npy file: {time.time()-t3:.2f} s")

    return 

if __name__ == "__main__":
     N = 80000
     generate_save_R_fc(N)
#     num_trials = 1
#     num_workers = 1
            
#     # create the array of shape (1 x num_trials) with value N at each index that will go as input in the pool.map
#     args = np.full(num_trials, N)
        
#     # call the generating function in parallel
        
#     #pool = Pool(processes=num_workers)
#     #list(tqdm.tqdm(pool.imap_unordered(generate_save_R_fc, args), total=num_trials))
#     with Pool(processes=num_workers) as pool:
#          results = pool.map(generate_save_R_fc, args)