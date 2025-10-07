#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:28:11 2024

@author: naumanibrahim
"""
import numpy as np
import scipy as sp
import time
import random
from numba import njit, prange 
from scipy.sparse import csr_matrix, lil_matrix, identity, save_npz, load_npz
import os
from os.path import join, expanduser
import glob
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool, set_start_method
import tqdm

def Minkowski_cube(rho = 10000, D = 4, l=0.2):
    code_start_time=time.time()
   
    ###define things that we need###
    #N is the total number of points
    N=rho
    height = 1.0
    #coords is the unsorted array of coordinates of the points in the hypercube
    coords = np.zeros((N,D)) # this creates an array with N rows and D columns
    
    #Force coding the two points of interest, a and b, at (1,-l/2,0,0) and (1,l/2,0,0)
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
    print(f"Generating the final coordinates took: {time.time()-code_start_time:.2f} s")
    return final_coords


# ######################## Generating a relations matrix ################################################

@njit(parallel=True)
def R_from_fc_numba(final_coords):
    N=len(final_coords)
    R = np.zeros((N,N)) # initialize the relations matrix
    
    for i in prange(N):
        t_i=final_coords[i,0] #labeling the coordinates of point i. For some reason doing it this way instead of 
        x_i=final_coords[i,1] #t_i, x_i, y_i, z_i = final_coords[i] is faster (19s vs 26s for N=10k). Also, 
        y_i=final_coords[i,2] #it is important to define them at this stage rather than in the second loop for speed!
        z_i=final_coords[i,3]
        
        for j in range(i+1,N): #the start is i+1 because of natural labeling j<=1 cannot be to the future of i
            t_j=final_coords[j,0] #labeling the coordinates of point j
            x_j=final_coords[j,1] 
            y_j=final_coords[j,2]
            z_j=final_coords[j,3]
            
                # Check the condition
            if (t_j - t_i)**2 > (x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2: #this condition says 
                #that if the timelike separation between j and i is more than the spacelike separation
                #then j is to the future of i.
                #if t_j-t_i > np.linalg.norm(final_coords[j,1:]-final_coords[i,1:]): This condition is slower!
                R[i,j]=1
            #   print("Found a relation!")
    return R

def R_from_fc_files(final_coords):
    t0 = time.time()
    R = R_from_fc_numba(final_coords)
    t1 = time.time()
    print(f"---Finding R took {t1 - t0: .2f} seconds---") 
    
    R_sparse = csr_matrix(R)
    t1 = time.time()
    print(f"Converting to csr_matrix took: {t1 - t0: .2f} s")
    
    return R_sparse

@njit(parallel=True)
def find_relations(final_coords):
    N = len(final_coords)
    
    # Estimate the maximum number of relations (upper bound for memory preallocation)
    max_relations = int(N * N/2)  # Adjust based on expected sparsity
    
    row_indices = np.empty(max_relations, dtype=np.int32)
    col_indices = np.empty(max_relations, dtype=np.int32)
    values = np.ones(max_relations, dtype=np.int8)  # All values will be 1

    count = 0  # Counter for actual relations found
    
    for i in prange(N):
        t_i, x_i, y_i, z_i = final_coords[i, 0], final_coords[i, 1], final_coords[i, 2], final_coords[i, 3]

        for j in range(i + 1, N):
            t_j, x_j, y_j, z_j = final_coords[j, 0], final_coords[j, 1], final_coords[j, 2], final_coords[j, 3]

            if (t_j - t_i) ** 2 > (x_j - x_i) ** 2 + (y_j - y_i) ** 2 + (z_j - z_i) ** 2:
                if count < max_relations:
                    row_indices[count] = i
                    col_indices[count] = j
                    count += 1

    return row_indices[:count], col_indices[:count], values[:count], N

def R_from_fc_sparse(final_coords):
    row_indices, col_indices, values, N = find_relations(final_coords)
    R_sparse = csr_matrix((values, (row_indices, col_indices)), shape=(N, N))
    return R_sparse


def causet_tau_c_files(folder_path, R_sparse, i, j):
    # R_sparse stored on RAM all the time.
    code_start = time.time()
    R_powerk_ij = 1
    k = 0
    N = R_sparse.shape[0]
    
    # Generate a CSR sparse identity matrix
    R_k = identity(N, format='csr') 
    
    while R_powerk_ij > 0: 
        # R_k in RAM as well at all times!
        
        k += 1      #putting it before makes the last run be the kth run
        # R_power=np.linalg.matrix_power(R,k)[c,a]
        #very slow code for large N
        
        #New attempt that is supposed to use already caluclated power to find the next one.
        t2 = time.time()
        
        R_k = R_sparse @ R_k
        # I think over here I should do R_k.data = np.ones_like(R_k.data) to make all values 1.
        
        t3 = time.time()
        
        # Print when multiplication complete
        print(f"R^{k} found and it took {t3 - t2: .2f} s")
        
        # Store R_k to npz file:
        file_name = f'R_{k}.npz'
        file_path = join(folder_path, file_name)
        save_npz(file_path, R_k)
        
        #we want to know how many maximal chains are there between c and a:
        numofmaxchains = R_powerk_ij
        
        R_powerk_ij = R_k[i,j] #If the ca-th element of R^k=0 then the longest chain is k-1 long
                                #this is because the power of R is the number of links in the chain
                                            #the fact that at k-th power R^k_pq = 0 means there are
                                            #no chains with k links, so the cardinality of the chain is k-1.
        
    t1 = time.time()
    print(f"Finding tau_c took: {t1 - code_start: .2f} s")
    print(f"There are {numofmaxchains} maximal chains of length {k-1} between element {i} and {j}")
    
    return k-1

def parallel_multiply(A, B):
    
    return

def causet_tau_c_files_parallel(folder_path, R_sparse, i, j):
    # R_sparse stored on RAM all the time.
    code_start = time.time()
    R_powerk_ij = 1
    k = 0
    N = R_sparse.shape[0]
    
    # Generate a CSR sparse identity matrix
    R_k = identity(N, format='csr') 
    
    while R_powerk_ij > 0: 
        # R_k in RAM as well at all times!
        
        k += 1      #putting it before makes the last run be the kth run
        # R_power=np.linalg.matrix_power(R,k)[c,a]
        #very slow code for large N
        
        #New attempt that is supposed to use already caluclated power to find the next one.
        t2 = time.time()
        
        R_k = R_sparse @ R_k
        #R_k = parallel_multiply(R_sparse, R_k)
        
        ###### I think over here I should do R_k.data = np.ones_like(R_k.data) to make all values 1.
        R_k.data = np.ones_like(R_k.data)
        
        t3 = time.time()
        
        # Print when multiplication complete
        print(f"R^{k} found and it took {t3 - t2: .2f} s")
        
        # Store R_k to npz file:
        file_name = f'R_{k}.npz'
        
        
        file_path = join(folder_path, file_name)
        save_npz(file_path, R_k)
        
        #we want to know how many maximal chains are there between c and a:
        #numofmaxchains = R_powerk_ij ## This will not work if I reset data to ones for each power
        
        R_powerk_ij = R_k[i,j] #If the ca-th element of R^k=0 then the longest chain is k-1 long
                                #this is because the power of R is the number of links in the chain
                                            #the fact that at k-th power R^k_pq = 0 means there are
                                            #no chains with k links, so the cardinality of the chain is k-1.
        
    t1 = time.time()
    print(f"Finding tau_c took: {t1 - code_start: .2f} s")
    #print(f"There are {numofmaxchains} maximal chains of length {k-1} between element {i} and {j}")
    
    return k-1
def L_from_R_files(folder_path):
    t00 = time.time()
    
    file_path_1 = join(folder_path, 'R_1.npz' )
    file_path_2 = join(folder_path, 'R_2.npz' )
    
    R_sparse =  load_npz(file_path_1)
    R_2 = load_npz(file_path_2)
    
    # Change all the data of R_2 to 1s
    R_2.data = np.ones_like(R_2.data) # if I want to use np.ones then I would first have to define the array size etc.
    
    L = R_sparse - R_2
    
    t1 = time.time()
    print(f"Total time to build L (optimized): {t1 - t00:.2f} s")
    
    return L

######################Finding LMCs########################33

####### LMCs_v3 is the final version #######
def LMCs_v32_files(folder_path, L, k_max, i, j):
    ######################## Finding powers of L ###############################
    t00 = time.time()

    #################### Create dictionary with a level structure ##############
    # Each level's key is the level number l, starting from k_max - (l = 1) until
    # l = k_max (k_max is the number of links in the LMCs. The number of elements, however, is
    # k_max + 1 thus, l  = k_max + 1 would contain only i and l = 0 would contain only j.) Each
    # level will have the elements at that level, and also the relationships with elements one
    # level up the chain. 
    
    ##### structure of all_levels: ############
    # Each level's key is the level number l. 
    # Each level has two keys: 'filtered_elements' and 'relationships'. The former's 'value' is a list of elements at that level, 
    # but the latter's 'value' is another dictionary whose 'keys' are the elements one level up and its 'values' are lists
    # of elements at the current level that are to the past of the elements one level up. In this sense each level's key contains both the 
    # elements at that level and those one level up. 
    
    all_levels = {}
    current_elements = [j]
    current_level = k_max
    all_levels[current_level] = {
        'filtered_elements': current_elements,
        'relationships': {element: [] for element in current_elements}
        }
    
    #################### starting the for loop over l for updating each level ##########
    
    for l in range(1,k_max): # slightly confused if it should be k_max + 1?
        next_level = current_level - 1
        filtered_elements = []
        relationships = {element: [] for element in current_elements}
        
        # Accessing the R^next_level th matrix using the matrix stored on disk
        file_name = f'R_{next_level}.npz'
        file_path = join(folder_path, file_name )
        Rtopower = load_npz(file_path)
        
        # Finding indices where the ith row is stored
        row_start = Rtopower.indptr[i]
        row_end = Rtopower.indptr[i+1]
        columns = Rtopower.indices[row_start:row_end]
        
        for candidate in columns:
            isrelated = False
            
            for oneofcurrentelement in current_elements:
                if L[candidate, oneofcurrentelement] == 1:
                    relationships[oneofcurrentelement].append(candidate)
                    
                    if not isrelated:
                        filtered_elements.append(candidate)
                        isrelated = True
                        
        all_levels[next_level] = {
            'filtered_elements': filtered_elements,
            'relationships': relationships
        }

        current_elements = filtered_elements
        current_level = next_level
        
    
    t1 = time.time()
    print(f"It takes {t1 - t00:.2f} s to find all the elements of LMCs")
    return all_levels


import pandas as pd
from openpyxl import load_workbook

# Function to save DataFrame to an Excel file
def save_to_excel(df, filename):
    if not os.path.exists(filename):
        df.to_excel(filename, index=False, sheet_name="Data")
    else:
        with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            book = load_workbook(filename)
            sheet = book.active
            start_row = sheet.max_row + 1  # Append new data below existing data
            df.to_excel(writer, index=False, header=False, startrow=start_row, sheet_name="Data")
############# Putting together all the functions ##############

def LMCs_files(N = 10000):
    #initialize other variables
    D, l, i, j = 4, 0.2, 0, N - 1
    
    t00 = time.time()
    parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "Saving Rd multiprocessing", "temp")
    # Create a unique folder for this process
    folder_name = f"temp_{os.getpid()}"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    fc = Minkowski_cube(N, D, l)
    R = R_from_fc_files(fc)
    k_max = causet_tau_c_files(folder_path, R, i, j)
    L = L_from_R_files(folder_path)
    all_levels = LMCs_v32_files(folder_path, L, k_max, i ,j)
    
    tf = time.time()
    print(f"---Total time to find LMCs for N = {N}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")
    
    # Remove the folder for this process
    os.rmdir(folder_path)
    
    return fc, all_levels


##### Visualizing the LMCs #########################3

def visualizing_LMCs(N):
    ##### structure of all_levels: ############
    # Each level's key is the level number l. 
    # Each level has two keys: 'filtered_elements' and 'relationships'. The former's 'value' is a list of elements at that level, 
    # but the latter's 'value' is another dictionary whose 'keys' are the elements one level up and its 'values' are lists
    # of elements at the current level that are to the past of the elements one level up. In this sense each level's key contains both the 
    # elements at that level and those one level up. 
    # level up the chain. The elements are stored as numbers which correspond to the index of the
    # coordinates for that element in fc.
    
    ##### The goal of visualizing for now is only to see the location of the all the points at 
    # each level - I am not interested in what elements they are linked to above and below.
    # I think the visualization should be in (t, r, z) coordinates and I should ignore the angular coordinate.

    # I will need to find the coordinates of the elements at each level using the fc (final_coords)
    # array. To do this I will have to save the final coordinates: I can add fc to the return object of LMCs.
    # Should I create a seperate dictionary to store the coordinates? Lets say for now that no. 
    # I should color-code the points based on their level though. The levels will grow with the 
    # t coordinate. 
    # 
    ###### My first task is to extract all the elements without, for now, caring about their level.
    # The link <https://www.geeksforgeeks.org/ways-to-extract-all-dictionary-values-python/> explains how to do this for a dictionary
    # with a simple structure. For a nested dictionary as mine, take inspiration from <https://stackoverflow.com/questions/40657403/how-to-extract-multi-level-dictionary-keys-values-in-python>
    # first use all_levels.items(). It returns tuples of key numbers and dictionaries, so:
    
        ########## Generating the fc and all_levels ###########
    
    #initialize other variables
    D, l, i, j = 4, 0, 0, N - 1
    
    t00 = time.time()
    parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "Saving Rd multiprocessing - R_sparse direct", "temp")
    # Create a unique folder for this process
    folder_name = f"temp_{os.getpid()}"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    fc = Minkowski_cube(N, D, l)
    R = R_from_fc_sparse(fc)
    k_max = causet_tau_c_files(folder_path, R, i, j)
    L = L_from_R_files(folder_path)
    all_levels = LMCs_v32_files(folder_path, L, k_max, i ,j)
    
    tf = time.time()
    print(f"---Total time to find LMCs for N = {N}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")
    
    # Remove all the tpz files from temp:
    files = glob.glob(folder_path+'/*')
    for f in files:
        os.remove(f)
    
    #####################################################3
    
    all_elements = []
    for l, sub_dict in all_levels.items():
        all_elements.append(sub_dict['filtered_elements']) # returns a list of lists starting with the filtered elements in the top-most layer.
    all_elements.reverse() # inverts the list ordering
    
    ###### Now I want to associate to each element its coordinates using fc
    # all_elements is a list of list. to iterate over it:
    all_points = []
    for level in all_elements:
        for element in level:
            all_points.append(fc[element])  # does not create a list of lists of lists as I had hoped.
                                            # creates a list of the coordinates of all the points.
    all_points = np.array(all_points) # converts a list of arrays to an array
    ###### I would like to convert these coordinates to (t, r) of spherical coordinates and ignore the theta and phi coordinate.
    # the r coordinate is given by sqrt(x**2 + y**2 + z**2).
    # all_points_spherical = np.column_stack((all_points[:, 0], np.sqrt(all_points[:, 1]**2 + all_points[:, 2]**2 + all_points[:, 3]**2)))
    t = all_points[:, 0]
    r = np.sqrt(all_points[:, 1]**2 + all_points[:, 2]**2 + all_points[:, 3]**2)
    x = all_points[:, 1]
    y = all_points[:, 2]
    z = all_points[:, 3]
    
    # Create DataFrame
    df = pd.DataFrame({'t': t, 'r': r, 'x': x, 'y': y, 'z': z})
    
    # saving to csv format
    df.to_csv(f"spherical_coordinates_N{N}_l_0.csv", mode='a', index=False, header=False)
    
    ##### saving the spherical coordinates to the same excel file
    
    # Create DataFrame
    #df = pd.DataFrame({'t': t, 'r': r, 'x': x, 'y': y, 'z': z})
    
    # Save to Excel with lock
    #save_to_excel(df, f"spherical_coordinates_N{N}_l_0.xlsx")
    
    
    # ###### Plotting all_points_spherical as a scatter plot using matplotlib.pyplot as plt
    
    # # polar r vs time
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.scatter(all_points_spherical[:,1], all_points_spherical[:,0], c='blue', marker='o')

    # # Add labels
    # ax1.set_ylabel('Time')
    # ax1.set_xlabel('Spherical r')
    
    # # Set plot title
    # ax1.set_title(f'r vs t for N = {N}')
    
    
    # # 3d scatter plot
    # #fig3 = plt.figure()
    # #ax3 = fig3.add_subplot(111, projection = '3d')
    # #ax3.scatter(all_points_polar[:,0], all_points_polar[:,1], all_points_polar[:,2], c='blue', marker='o')
    
    # # Add labels
    # #ax3.set_ylabel('z')
    # #ax3.set_xlabel('Polar r')
    # #ax3.set_zlabel('z')
    
    # # Show the plots
    # plt.show()
    # #save figures
    # folder_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "Saving Rd", "rt pngs")
    # file_name = f"rt_N{N}.png"
    # fig1.savefig(join(folder_path, file_name))
    
    return 

##### implementing parralelization for running 10 trials ######
## The current implementation only every creates the the same no. of 
# folders as num_workers in which to store the powers of R. This means 
# the R_{k} files are rewritten whenever a woker starts on a second running
# of the code. It could have caused problems if the powers were used before 
# all of them were generated for tau_c, so for now its safe. It is possible
# for the second run to produce smaller maximum power of R than the first
# run, so the temp folder can contain the last few pwers of the previous run.
# These powers are not called in the second run however, because LMCs_v32_files()
# taken k_max as an argument and only finds levels depending on the value of K_max.

#if __name__ == "__main__":
    
#    set_start_method('fork')
#    N = 80000
#    num_trials = 3
#    num_workers = 1
        
    # create the array that will go as input in the pool.map
#    args = np.full(num_trials, N)
    
    # call the vizualization in  parallel
    
#    pool = Pool(processes=num_workers)
#    list(tqdm.tqdm(pool.imap_unordered(visualizing_LMCs, args), total=num_trials))
      
visualizing_LMCs(80000)

















