#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:28:11 2024

Changing graph storage from csr arrays to dense bool type 2D array
@author: naumanibrahim
"""
import numpy as np
import time
import random
from numba import njit, prange, set_num_threads, types
from numba.typed import List
#from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from os.path import join, expanduser
import glob
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool, set_start_method
import tqdm
import networkx as nx
from collections import defaultdict, deque
import pandas as pd
import h5py
import json
from memory_profiler import profile

def Minkowski_interval(interval_size=10000, height=1, dimension=4):
    N = interval_size + 2
    coords = np.zeros((N,dimension))
    coords[N-1,0] = height
    success = 1
    coord = np.zeros((1,dimension))
    while success < N - 1:
        coord[0,0] = height * random.random()
        coord[0,1:] =  height * np.random.rand(1,dimension-1) - 0.5 * height
        if coord[0,0] < 0.5 * height:
            width = coord[0,0]
        else:
            width = height - coord[0,0]
        if np.linalg.norm(coord[0,1:]) > width:
            continue
        coords[success,:] = coord[0,:]
        success += 1
        # print(coord[0,0],coord[0,1])
    srt_coords = coords[np.argsort(coords[:, 0])]
    return srt_coords

#@njit
def Minkowski_cube(rho = 10000, height = 1, D = 4, l=0):
    #code_start_time=time.time()
   
    ###define things that we need###
    #N is the total number of points
    N = rho
    width = 1
    #coords is the unsorted array of coordinates of the points in the hypercube
    coords = np.zeros((N,D)) # this creates an array with N rows and D columns
    
    #Force coding the two points of interest, a and b, at (1,-l/2,0,0) and (1,l/2,0,0)
    coords[0,:]=[height,-l/2,0,0]
    #coords[1,:]=(1,l/2,0,0) # for now I want l = 0 and I don't want there to be to elements there
    #and also the coordinates for c, the comparison point to be at (0,0,0,0):
    coords[1,:]=[0,0,0,0]
    #coords[N-1,0] = height
    #trial_coord are the coordinates of one point. It will be generated repeatedly
    temp_coord = np.zeros((1,D)) #defining and initializing the temp coord as an array of
    #a single row with D columns filled with zeros. two braces are necessary in using np.zeros. 
    #np.zeros((1)) the same as np.zeros((1,1)).
    successes = 2 #defining the counter that tells when the coords array is full
    
    while successes < N: #colon is need for loops apparently
     
    #generate temp_coord where the domain of the cube is [0,1] in time and [-0.5,0.5] for x, y, z.
        temp_coord[0,0] = height * random.random() #creates the time coordinate
        temp_coord[0,1:] = width * np.random.rand(1,D-1) - width * 0.5 # this simultaneously
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
    #print(f"Generating the final coordinates took: {time.time()-code_start_time:.2f} s")
    return final_coords

######################## DiGraph with negative weights for reference ###################
def iter_weighted_edges_from_neighbors(Rp, N, weight=-1):
    for u in range(N):
        nbrs = get_row_neighbors(Rp, N, u)
        for v in nbrs:
            yield (u, v, weight)

def graph_from_neighbors_weighted(Rp, N, weight=-1):
    t0 = time.time()
    G = nx.DiGraph()
    G.add_weighted_edges_from(iter_weighted_edges_from_neighbors(Rp, N, weight))
    print(f"Created weighted DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges "
          f"in {time.time() - t0:.2f}s")
    return G

####################### Finding LMCs using NetworkX ################

## To get a networkx digraph ## I wonder if giving it R and asking it to find L using transitive reduction would be faster?
# G = nx.from_scipy_sparse_array(csr_matrix(L), create_using=nx.DiGraph)


## Find the longest path between two nodes
# longest_path = nx.dag_longest_path(G, source=start_node, target=end_node)
# This will return the list of nodes in the overall longest path in the DAG.

## Finding LMCs using R to create G, then G_neg, and finally, all_shortest_paths
def all_levels_from_G_neg(G_neg, source, target):
    t0 = time.time()
    LMCs = list(nx.all_shortest_paths(G_neg, source, target, weight="weight", method="bellman-ford"))
    t1 = time.time()
    print(f"Time taken to find all the LMCs using a DiGraph: {t1 - t0:.2f} s")
    
    def unique_levels(paths):
        """
        paths: list of lists, all same length
        returns: list of sets, each set = unique elements at that level
        """
        if not paths:
            return []
    
        path_len = len(paths[0])
        levels = [set() for _ in range(path_len)]
    
        for path in paths:
            for i, node in enumerate(path): # produces pairs of (i, element_j_of_list_i) with j in the outer loop and i in the inner loop.
                levels[i].add(node)
    
        # convert each set to sorted list
        return [sorted(s) for s in levels]
    
    levels = unique_levels(LMCs)
    return levels



# --------- layout helpers (strict upper triangle, row-major) ---------

@njit(inline='always')
def start_of_row(i, N):
    # number of bits before row i: sum_{k=0}^{i-1} (N-1-k)
    # = i*(2N - i - 1)/2
    return (i * (2*N - i - 1)) // 2

@njit(inline='always')
def bit_index(i, j, N):
    # require 0 <= i < j < N
    return start_of_row(i, N) + (j - i - 1)

@njit(inline='always')
def total_bits_upper(N):
    return N*(N-1)//2

@njit(inline='always')
def total_bytes_upper(N):
    tb = total_bits_upper(N)
    return (tb + 7) // 8

# --------- allocate ---------

@njit
def make_empty_packed_upper(N):
    """Allocate 1D packed upper-triangle bit array (uint8)."""
    return np.zeros(total_bytes_upper(N), dtype=np.uint8)

# --------- set / clear / get single (i,j) with i<j ---------

@njit
def set_edge(P, N, i, j):
    if not (0 <= i < j < N):
        raise IndexError
    idx = bit_index(i, j, N)              # global bit index
    
    # 1. Which byte should j go to in row i 
    b   = idx >> 3                        # byte index. 
                                # same as j // 8, i.e., remainder without decimal. 
                                # >> is a bitwise operator that removes bits from the end
                                # x >> y is equivalent to x / 2**y
    
    # 2. Which bit should it go to
    k   = idx & 7                         # bit position 0..7 (LSB-first)
                                # j & 7 == j mod 8, gives the remainder when dividing by 8. MSB (most significant bit) - first within each byte
    
    # 3. Update the byte in the right bit by using bitwise OR
    # We use MSB-first inside each byte:
    mask = np.uint8(1 << (7 - k))
                                # Rp[i, byte_idx] = Rp[i, byte_idx] | (1 << bit_pos) where | is a bit-wise OR addition
    P[b] |= mask
    
@njit
def clear_edge(P, N, i, j):
    if not (0 <= i < j < N):
        raise IndexError
    idx = bit_index(i, j, N)
    b   = idx >> 3
    k   = idx & 7
    mask = np.uint8(1 << (7 - k))
    P[b] &= np.uint8(~mask)

@njit
def get_edge(P, N, i, j):
    if not (0 <= i < j < N):
        raise IndexError
    idx = bit_index(i, j, N)
    b   = idx >> 3
    k   = idx & 7
    return int((P[b] >> np.uint8(7 - k)) & np.uint8(1))

# --------- read a whole row i (neighbors j>i) ---------

@njit
def get_row_neighbors(P, N, i):
    L = N - 1 - i
    out = List.empty_list(types.int64)
    if L <= 0:
        return out

    r0   = start_of_row(i, N)
    rEnd = r0 + L
    b0   = r0 >> 3
    bEnd = (rEnd + 7) >> 3

    for b in range(b0, bEnd):
        byte = P[b]
        if byte == 0:
            continue
        byte_msbit = b << 3
        # Just scan all 8 bits; guard with g<rEnd
        for bit in range(8):
            g = byte_msbit + bit
            if g < r0 or g >= rEnd:
                continue
            # MSB-first test
            if (byte >> (7 - bit)) & 1:
                j = i + 1 + (g - r0)
                if j < N:      # redundant but harmless safety
                    out.append(j)
    return out

# Set number of threads to be used by numba. Might be important for running on cluster
# set_num_threads(8)

@njit(parallel=True) # For N = 5k time goes down from 5s --> 0.09s
def R_packed(final_coords):
    N = len(final_coords)
    
    # Initialize R
    R_packed = make_empty_packed_upper(N)  
    num_edges_E = 0
    
    for i in prange(N): 
        t_i, x_i, y_i, z_i = final_coords[i]
        
        for j in range(i + 1, N):
            t_j, x_j, y_j, z_j = final_coords[j, 0], final_coords[j, 1], final_coords[j, 2], final_coords[j, 3]

            if (t_j - t_i)**2 > (x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2:
                set_edge(R_packed, N, i, j)
                num_edges_E += 1
    return R_packed, num_edges_E

############ Finding LMCs using edge relaxation (R_packed + Numba) #######

@njit # Poduces: pred(~n x max_pred), pred_count and distance arrays
def find_longest_paths_numba_R_packed(n, R_packed, s, t, max_pred=50):
    """
    Find all longest paths from source (s) to target (t) in a DAG using
    Bellman-Ford with negative weights
    
    Parameters:
    - n: Number of nodes
    - R_packed: Relations matrix as a boolean 2D array to save memory. but packed.
    - s: Source node
    - t: Target node
    - max_pred: Maximum number of predecessors per node (for array allocation)
    
    Returns:
    - distance, pred, pred_count
    """
    # Initialize distance array with a large integer value (infinity)
    distance = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    distance[s] = 0  # Set source distance to 0
    
    # Initialize predecessors: 2D array (n, max_pred) and count array
    pred = np.full((n, max_pred), -1, dtype=np.int32)  # -1 means no predecessor
    pred_count = np.zeros(n, dtype=np.int32)  # Number of predecessors per node
    
    # Relaxation step: Iterate over all nodes
    for u in range(n):
        if distance[u] != np.iinfo(np.int64).max:  # If node u is reachable, i.e. is in the future of s
            
            #row_u = np.unpackbits(R_packed[u], bitorder='big')[:n]
            adj_u = get_row_neighbors(R_packed, n, u) # creates an int32 array containing indices of non_zero entries in row[u] of R_bool 
            for v in adj_u: # v is a neighbor node of u
                new_dist = distance[u] - 1  # Assume edge weight is -1 for longest path
                if distance[v] == np.iinfo(np.int64).max or new_dist < distance[v]:
                    # Found a better (longer) path
                    distance[v] = new_dist
                    pred[v, 0] = u  # Reset predecessors
                    pred_count[v] = 1
                elif new_dist == distance[v]:
                    # Found an equally long path
                    if pred_count[v] < max_pred:
                        pred[v, pred_count[v]] = u  # Add predecessor
                        pred_count[v] += 1
                    else:
                        raise ValueError("max_pred not big enough!")
    return distance, pred, pred_count

@njit # Finds all the nodes on the LMCs as a boolean array called visited(pred.shape[0]) with True values for the correct nodes
def find_nodes_on_longest_paths_numba(S, T, pred, pred_count, n):
    """
    Find all unique nodes on any longest path from S to T in a DAG.

    Args:
        S (int): Source node index
        T (int): Target node index
        pred (array): 2D array where pred[u, i] is a predecessor of u
        pred_count (array): 1D array where pred_count[u] is the number of predecessors of u
        n (int): Number of nodes

    Returns:
        array: Boolean array where True indicates a node on some longest path from S to T
    """
    visited = np.zeros(n, dtype=np.bool_)   # Zeros are interpreted as False!
                                            # Using a visited array for all the nodes of the causal set 
                                            # means that unlike creating all_levels directlty, one does not 
                                            # have to check if the visited element is already in all_levels.
    queue = [T]
    visited[T] = True
    while queue:
        current = queue.pop(0) # Removes elements from the start of the queue. 
                                # Makes it a BFS. Changing to pop() probably changes it to a DFS!
        for i in range(pred_count[current]):
            u = pred[current, i]
            if not visited[u]:
                visited[u] = True 
                queue.append(u) # Adds elements to the end of the queue.
    return visited

# Python function to form list of lists
def find_nodes_by_level_numba(S, T, visited, distance, n):
    """
    Organize nodes on longest paths by their distance from S.

    Args:
        S (int): Source node index
        T (int): Target node index
        pred (array): 2D array where pred[u, i] is a predecessor of u
        pred_count (array): 1D array where pred_count[u] is the number of predecessors of u
        distance (array): 1D array where distance[u] is the distance from S to u
        n (int): Number of nodes

    Returns:
        list: List of lists where list at index i contains nodes at distance i from S
    """
    if distance[T] == np.iinfo(np.int32).max:
        return []  # No path from S to T
    max_dist = -distance[T]  # Number of edges in longest path from S to T
    levels = [[] for _ in range(max_dist + 1)]
    
    for u in range(n):
        if visited[u]:
            level = -distance[u]  # Convert negative distance to level
            levels[level].append(u)
    for sublist in levels: # sort in ascending order
        sublist.sort()
    return levels

# Python function to combine the previous results
def numpy_R_packed_LMCs(n, R_packed, s, t, max_pred):
    t0 = time.time()
    # Find the pred, pred_count and distance arrays
    distance, pred, pred_count = find_longest_paths_numba_R_packed(n, R_packed, s, t, max_pred)
    t1 = time.time()
    #print(f"Time taken to make pred arrays using numba: {t1 - t0:.2f} s")
    
    # Find the nodes on the LMCs
    visited = find_nodes_on_longest_paths_numba(s, t, pred, pred_count, n)
    t2 = time.time()
    #print(f"Time taken to make visited array using numba: {t2 - t1:.2f} s")

    # Find the all_levels array
    all_levels = find_nodes_by_level_numba(s, t, visited, distance, n)
    t3 = time.time()
    #print(f"Time taken to make all_levels: {t3 - t2:.2f} s")

    return all_levels, pred, pred_count

####################### Visualizing the LMCs #########################
#@profile
def Visualizing_LMCs_R_packed_interval(interval_size, height, D):
    
    ##### The goal of visualizing for now is only to see the location of the all the points at 
    # each level - I am not interested in what elements they are linked to above and below.
    # I think the visualization should be in (t, r) coordinates and I should ignore the angular coordinate in the final analysis but include them in the files.

    # I will need to find the coordinates of the elements at each level using the fc (final_coords)
    # array. To do this I will have to save the final coordinates: I can add fc to the return object of LMCs.
    # Should I create a seperate dictionary to store the coordinates? Lets say for now that no. 
    # I should color-code the points based on their level though. The levels will grow with the 
    # t coordinate. 
    
    ###### My first task is to extract all the elements without, for now, caring about their level.
    # The link <https://www.geeksforgeeks.org/ways-to-extract-all-dictionary-values-python/> explains how to do this for a dictionary
    # with a simple structure. For a nested dictionary as mine, take inspiration from <https://stackoverflow.com/questions/40657403/how-to-extract-multi-level-dictionary-keys-values-in-python>
    # first use all_levels.items(). It returns tuples of key numbers and dictionaries, so:
    
        ########## Generating the fc and all_levels ###########
    
    #initialize other variables
    N = interval_size + 2
    i, j = 0, N - 1
    max_pred = 250
    t00 = time.time()
    
    # Create fc
    t00 = time.time()
    fc = Minkowski_interval(interval_size, height, D) # produces fc with interval_size+2 coordinates
    
    # Create rows and columns for the edges
    t0 = time.time()
    #print(f'Time to make fc: {t0 - t00:.2f}s')
    R, num_edges_E = R_packed(fc)
    #print(f"Making R_packed took: {time.time() - t0: .2f} s")
    #print(f"The number of relations, E, for N = {N/1000:.0f}k is: E = {num_edges_E:.2e}")
    
    # Find all_levels
    all_levels, pred, pred_count = numpy_R_packed_LMCs(N, R, i, j, max_pred)
    tf = time.time()
    #print(f"---Total time to find LMCs for N = {N}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")
    
    
    # ###### Now I want to associate to each element its coordinates using fc
    # all_levels is a list of list. to iterate over it:
    all_points = []
    for level, elements in enumerate(all_levels):
        for element in elements: 
            t, x, y, z = fc[element]
            
            # find the preds of the element and store in list
            preds = pred[element, 0:pred_count[element]]
            
            r = np.sqrt(x**2 + y**2 + z**2)
            if r == 0:
                theta = 0
                phi = 0
            else:
                theta = np.arctan2(y ,x)
                phi = np.arccos(z/r)
            
            # all_points.append([element, level, t, r, theta, phi, preds, interval_size, height, D]) 
            all_points.append([element, level, t, r, theta, phi]) 
    
    return all_points

def _helper(params):
    N, height, D = params
    return Visualizing_LMCs_R_packed_interval(N, height, D)

import csv

if __name__ == "__main__":
    # Settings
    height = 10
    D = 4
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos",
                        "cst_longest_maximal_chains", "DAGs", "LMCs_data", "Intervals")
    folder_name = f"Height {height}"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Ns = [26180, 39270, 52360, 74048, 104720, 148096, 209440, 296193, 418880, 592368, 1184736]
    # Ns = [592386]
    # Ns = [296193]
    # Ns = [74048, 148096]
    # Ns = [654,  1309,  2618,  3927,  5236,  6545,  7854,  9163, 10472, 13090, 15708, 18326, 20944]
    Ns = [654]
    num_trials = 1

    for N in Ns:
        # Determine number of worker processes per N
        if N >= 420000:
            num_workers = 1
            num_trials = 10
        elif N >= 290000:
            num_workers = 4
            num_trials = 40
        elif N >= 160000:
            num_workers = 9
        else:
            num_workers = 12
        
        file_name = f"spherical_coordinates_N{int(N/1000)}k.csv"
        out_path = join(folder_path, file_name)

        # Open output file and write header. "a" ensure appending to file, not overwrite
        with open(out_path, "a", newline="") as fout:
            writer = csv.writer(fout)
          
            # write header only if file is empty
            if fout.tell() == 0:
                # writer.writerow(["element", "level", "t", "r", "theta", "phi", "preds", "Interval size", "Height", "D" ])
                writer.writerow(["element", "level", "t", "r", "theta", "phi"])
 

            # Spawn pool of workers
            with Pool(processes=num_workers) as pool:
                # args: each worker gets the same N, repeated num_trials times
                args = [(N, height, D)] * num_trials
                start_time = time.perf_counter()
                counter = 0
                for rows in pool.imap_unordered(_helper, args):
                    writer.writerows(rows)
                    counter += 1
                    if counter % num_workers == 0: 
                        print(f"Finished {counter} sprinklings for N={int(N/1000)}k in {time.perf_counter() - start_time:.2f}s")
        print(f"Finished writing CSV for N={N} with {num_workers} workers.")

# Unix command to run on biggee so that I can log out:
    # nohup python3 myscript.py > output.txt 2>&1 &

# Unix command to copy file from biggee to Desktop
    # scp sibrahim@biggee.phy.olemiss.edu:/localhome/sibrahim/LMCs_data/Intervals/file.csv ~/Desktop/
    
    
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

# if __name__ == "__main__":
    
#     for N in [10000, 20000, 40000, 80000, 160000, 320000]: 
#         # if N != 320000 and N != 160000:
#         #     num_workers = 10
#         # if N == 160000:
#         #     num_workers = 6
#         # if N == 320000:
#         #     num_workers = 2
        
#         num_trials = 100
#         num_workers = 1
#         #create the array that will go as input in the pool.map
#         args = np.full(num_trials, N)
    
#         # call the vizualization in  parallel
#         pool = Pool(processes=num_workers)
#         list(tqdm.tqdm(pool.imap_unordered(Visualizing_LMCs_DAG_R_packed, args), total=num_trials))




# import tracemalloc # To track aximum memory requested by python

# tracemalloc.start()

# Visualizing_LMCs_DAG_R_packed(200000)

# current, peak = tracemalloc.get_traced_memory()
# print(f"Memory allocation peak: {peak/1024**2:.2f} MB")
# tracemalloc.stop()















