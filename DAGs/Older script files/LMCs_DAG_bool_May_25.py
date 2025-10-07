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

def Minkowski_interval(interval_size, height, dimension):
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


######################### Generating R_bool ################################################
########### a bool type is stored as 1byte. To store at the bit level we will have to contrive
@njit
def make_empty_packed(N):
    # Ceil division for bytes-per-row
    W = (N + 7) // 8
    return np.zeros((N, W), dtype=np.uint8)

@njit
def set_edge(Rp, i, j):
    # 1. Which byte should j go to in row i
    byte_idx = j >> 3           # same as j // 8, i.e., remainder without decimal. 
                                # >> is a bitwise operator that removes bits from the end
                                # x >> y is equivalent to x / 2**y
    # 2. Which bit should it go to
    bit_pos  = 7 - (j & 7)      # j & 7 == j mod 8, gives the remainder when dividing by 8. MSB (most significant bit) - first within each byte
    
    # 3. Update the byte in the right bit by using bitwise OR
    Rp[i, byte_idx] |= (1 << bit_pos) # Rp[i, byte_idx] = Rp[i, byte_idx] | (1 << bit_pos) where | is a bit-wise OR addition

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

@njit(parallel=True) # For N = 5k time goes down from 5s --> 0.09s
def R_packed(final_coords):
    N = len(final_coords)
    
    # Initialize R
    R_packed = make_empty_packed(N)  
    num_edges_E = 0
    
    for i in prange(N): 
        t_i, x_i, y_i, z_i = final_coords[i]
        
        for j in range(i + 1, N):
            t_j, x_j, y_j, z_j = final_coords[j, 0], final_coords[j, 1], final_coords[j, 2], final_coords[j, 3]

            if (t_j - t_i)**2 > (x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2:
                set_edge(R_packed, i, j)
                num_edges_E += 1
    return R_packed, num_edges_E


######################## DiGraph with negative weights for reference ###################
def DAG_neg_from_arrays(row_ptr, col_indices):
    t00 = time.time()
    # create row_indices from row_ptr
    num_rows = len(row_ptr) - 1
    num_elements = len(col_indices)
    row_indices = np.zeros(num_elements, dtype=int)

    for i in range(num_rows):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        row_indices[start:end] = i

    #create negative G
    G_neg = nx.DiGraph()
    
    # Add edges with weight -1 directly without constructing a matrix
    edges = zip(row_indices, col_indices)
    
    G_neg.add_weighted_edges_from((u, v, -1) for u, v in edges)
    print(f"Time taken to create G_neg: {time.time() - t00:.2f} s")

    return G_neg
####################### Finding LMCs using NetworkX ################

## To get a networkx digraph ## I wonder if giving it R and asking it to find L using transitive reduction would be faster?
# G = nx.from_scipy_sparse_array(csr_matrix(L), create_using=nx.DiGraph)


## Find the longest path between two nodes
# longest_path = nx.dag_longest_path(G, source=start_node, target=end_node)
# This will return the list of nodes in the overall longest path in the DAG.

## Finding LMCs using R to create G, then G_neg, and finally, all_shortest_paths
def LMCs_from_G_neg(G_neg, source, target):
    t0 = time.time()
    LMCs = list(nx.all_shortest_paths(G_neg, source, target, weight="weight", method="bellman-ford"))
    t1 = time.time()
    print(f"Time taken to find all the LMCs using a DiGraph: {t1 - t0:.2f} s")
    return LMCs

############ Finding LMCs using edge relaxation (Dictionaries) #######
# Method 1 using dictionaries
def find_longest_paths(n, row_ptr, col_indices, s, t):
    """
    Find all longest paths from source (s) to target (t) in a DAG using
    Bellman-Ford with negative weights.
    
    Parameters:
    - n: Number of nodes
    - edges: List of (u, v) pairs representing directed edges
    - s: Source node
    - t: Target node
    
    Returns:
    - List of longest paths from s to t
    """
    
    # Step 1: Supposes a topological order on row_ptr
    t1 = time.time() 
    
    # Step 2: Initialize distances and predecessors
    distance = {s: 0}  # Stores distance of nodes from s. Start with source at distance 0
    pred = defaultdict(list)  # Store predecessors
    
    ## Step 3: Bellman-Ford-like relaxation with negative weights
    # for u in col_indices[row_ptr[s]:row_ptr[s+1]]:
    for u in range(n):      # Process in topological order
        if u in distance :  # Only process reachable nodes of s 
                            # Need to figure out how to ignore nodes with no future relations
        
        # Access the adjacency list of u directly  
            for v in col_indices[row_ptr[u]:row_ptr[u+1]]:
                    new_dist = distance[u] + 1  # Treat edges as weight -1
                    if v not in distance or new_dist > distance[v]:  # not in distance is important for the first pass u = s, later it becomes redundant
                        distance[v] = new_dist
                        pred[v] = [u]  # Reset predecessors. This steps takes care of the fact that we have the relations rather than the links
                    elif new_dist == distance[v]:
                        pred[v].append(u)  # Append alternative predecessor
    t2 = time.time()
    print(f"Time taken to make the pred dictionary: {t2 - t1: .2f} s")

    # Step 4: Backtrack to reconstruct all longest paths
    # Recursive Method to give paths
    def backtrack(path, node):
        if node == s:
            all_paths.append(path[::-1])  # Reverse path
            return
        for parent in pred[node]:
            backtrack(path + [parent], parent)
    
    # Recursive Method to give levels
    def backtrack_2(level, all_levels):
        if level == 0:
            all_levels[level].append(t)
        
        for child in all_levels[level]:
            if child == s:
                all_levels[:] = all_levels[::-1]
                return True
            
        if len(all_levels) == level + 1:
            all_levels.append([])
        
        for child in all_levels[level]:
            for parent in pred[child]:
                if parent not in all_levels[level + 1]:
                    all_levels[level + 1].append(parent)   
        
        if backtrack_2(level + 1, all_levels):
            return True
    
    # Iterative Method to give lavels in two steps
    # first, find all the nodes along the LMCs
    # second, compare those nodes through the distance dictionary
    
    # First step
    def find_nodes_on_longest_paths(S, T, predecessors):
        """
        Find all unique nodes on any longest path from S to T in a DAG.

        Args:
            S (str): Source node
            T (str): Target node
            predecessors (dict): Dictionary where predecessors[V] is a list of nodes U
                             such that U -> V is an edge in the DAG

        Returns:s
            set: Set of nodes on any longest path from S to T, or empty set if no path exists
            """
        # Backward BFS from T
        visited = set()  # Store unique nodes
        queue = deque([T])
        visited.add(T)

        while queue:
            current = queue.popleft()
            # If current node has predecessors, explore them
            for pred in predecessors[current]:
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
                    
        return visited
    
    # Second step
    def find_nodes_by_level(S, T, predecessors, distances):
        """
        Organize nodes on longest paths by their distance from S.
        
        Args:
            S, T, predecessors: Same as above
            distances (dict): Distance from S to each node

        Returns:
            List of lists: indices are distances (0 to 14), values are lists of nodes at that level
    """
        nodes = find_nodes_on_longest_paths(S, T, predecessors)
        levels = [[] for _ in range(distances[t] + 1)]
        for node in nodes:
            levels[distances[node]].append(node)
        return levels
    
    ################## Calling path reconstruction algos ###############
    # Recursive
    # all_levels_recur = [[]]
    # backtrack_2(0, all_levels_recur)
    
    #all_paths = []
    #backtrack([t], t)
    
    
    # Iterative
    all_levels_iter = find_nodes_by_level(s, t, pred, distance)
    for sublist in all_levels_iter: # sort in ascending order
        sublist.sort()
    #print(f"---Time to find all_levels_iter for N={N}: {time.time() - t2: .2f} s")

    #return all_levels_recur, all_levels_iter
    return all_levels_iter
    #return distance, pred, all_levels_iter, all_paths

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
            adj_u = get_row_neighbors(R_packed, u, n) # creates an int32 array containing indices of non_zero entries in row[u] of R_bool 
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

    return all_levels

####### Running the program with multiple options commented out ######

# if __name__== "__main__":
    
#     N = 80000;
#     fc = Minkowski_cube(N, 4, 0)
    
#     t0 = time.time()
#     row_indices, col_indices = Edges_csr_arrays(fc)
#     print(f"Finding the arrays for the edges took: {time.time() - t0: .2f} s")
    
#     # G_rel_neg = DAG_neg_from_arrays(row_indices, col_indices)
#     # LMCs = LMCs_from_G_neg(G_rel_neg, 0, N-1)
    
#     #all_levels_recur, all_levels_iter = find_longest_paths(N, row_indices, col_indices, 0, N-1)
#     #distance, pred, all_levels_iter, all_paths = find_longest_paths(N, row_indices, col_indices, 0, N-1)
#     ###Calling Method 1
#     #all_levels_iter = find_longest_paths(N, row_indices, col_indices, 0, N-1)
#     #max_size = max(len(v) for v in pred.values())
    
#     max_pred = 300
#     all_levels_numba = numpy_LMCs(N, row_indices, col_indices, 0, N-1, max_pred)

    # For Parallelization
    #    num_trials = 3
    #    num_workers = 1
            
        # create the array that will go as input in the pool.map
    #    args = np.full(num_trials, N)
        
        # call the vizualization in  parallel
        
    #    pool = Pool(processes=num_workers)
    #    list(tqdm.tqdm(pool.imap_unordered(visualizing_LMCs, args), total=num_trials))

####################### Visualizing the LMCs #########################
#@profile
def Visualizing_LMCs_DAG_R_packed(N, height):
    
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
    D, l, i, j = 4, 0, 0, N - 1
    max_pred = 250
    t00 = time.time()
    
    # Create fc
    t00 = time.time()
    fc = Minkowski_cube(N, height, D, l)
    
    # Create rows and columns for the edges
    t0 = time.time()
    #print(f'Time to make fc: {t0 - t00:.2f}s')
    R, num_edges_E = R_packed(fc)
    #print(f"Making R_packed took: {time.time() - t0: .2f} s")
    #print(f"The number of relations, E, for N = {N/1000:.0f}k is: E = {num_edges_E:.2e}")
    
    # Find all_levels
    all_levels = numpy_R_packed_LMCs(N, R, i, j, max_pred)
    tf = time.time()
    #print(f"---Total time to find LMCs for N = {N}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")
    
    
    # ###### Now I want to associate to each element its coordinates using fc
    # all_levels is a list of list. to iterate over it:
    all_points = []
    for level, elements in enumerate(all_levels):
        for element in elements: 
            t, x, y, z = fc[element]
            r = np.sqrt(x**2 + y**2 + z**2)
            if r == 0:
                theta = 0
                phi = 0
            else:
                theta = np.arctan2(y ,x)
                phi = np.arccos(z/r)
            all_points.append([element, level, t, r, theta, phi]) 
           
    # all_points = np.array(all_points) # converts a list of arrays to an array
    # ###### I would like to convert these coordinates to (t, r) of spherical coordinates and ignore the theta and phi coordinate.
    # # the r coordinate is given by sqrt(x**2 + y**2 + z**2).
    # # all_points_spherical = np.column_stack((all_points[:, 0], np.sqrt(all_points[:, 1]**2 + all_points[:, 2]**2 + all_points[:, 3]**2)))
    # t = all_points[:, 0]
    # r = np.sqrt(all_points[:, 1]**2 + all_points[:, 2]**2 + all_points[:, 3]**2)
    # x = all_points[:, 1]
    # y = all_points[:, 2]
    # z = all_points[:, 3]
    
    # # Create DataFrame
    # df = pd.DataFrame(all_points)
    
    # # saving to csv format
    # df.to_csv(file_path, mode='a', index=False, header=False)
    
    return all_points

def _helper(params):
    N, height = params
    return Visualizing_LMCs_DAG_R_packed(N, height)

# import csv

# if __name__ == "__main__":
#     # Settings
#     height = 125
#     parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos",
#                         "cst_longest_maximal_chains", "DAGs", "LMCs_data")
#     folder_name = f"Height {height}"
#     folder_path = join(parent_path, folder_name)
#     os.makedirs(folder_path, exist_ok=True)

#     Ns = [160000]
#     num_trials = 100

#     for N in Ns:
#         # Determine number of worker processes per N
#         if N >= 320000:
#             num_workers = 2
#             num_trials = 40
#         elif N >= 160000:
#             num_workers = 6
#         else:
#             num_workers = 10
        
#         file_name = f"spherical_coordinates_N{int(N/1000)}k.csv"
#         out_path = join(folder_path, file_name)

#         # Open output file and write header once
#         with open(out_path, "w", newline="") as fout:
#             writer = csv.writer(fout)
#             writer.writerow(["element", "level", "t", "r", "theta", "phi"])

#             # Spawn pool of workers
#             with Pool(processes=num_workers) as pool:
#                 # args: each worker gets the same N, repeated num_trials times
#                 args = [(N, height)] * num_trials
#                 start_time = time.perf_counter()
#                 counter = 0
#                 for rows in pool.imap_unordered(_helper, args):
#                     writer.writerows(rows)
#                     counter += 1
#                     if counter % 10 == 0: 
#                         print(f"Finished {counter} sprinklings for N={int(N/1000)}k in {time.perf_counter() - start_time:.2f}s")
#         print(f"Finished writing CSV for N={N} with {num_workers} workers.")


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


###### How T / l_0 changes with N and T?
# for T in [1, 10, 100, 500, 1000]:
#     for n in range(6):
#         N = 10000 * 2 ** n
#         l_0 = (T/N) ** 0.25
#         print(f'{T}      {N}      {l_0:.2f}     {T/l_0:.2f}')



# T       N         l_0      T/l_0
# 1      10000      0.10     10.00
# 1      20000      0.08     11.89
# 1      40000      0.07     14.14
# 1      80000      0.06     16.82
# 1      160000      0.05     20.00
# 1      320000      0.04     23.78

# 10      10000      0.18     56.23
# 10      20000      0.15     66.87
# 10      40000      0.13     79.53
# 10      80000      0.11     94.57
# 10      160000      0.09     112.47
# 10      320000      0.07     133.75

# 100      10000      0.32     316.23
# 100      20000      0.27     376.06
# 100      40000      0.22     447.21
# 100      80000      0.19     531.83
# 100      160000      0.16     632.46
# 100      320000      0.13     752.12

# 500      10000      0.47     1057.37
# 500      20000      0.40     1257.43
# 500      40000      0.33     1495.35
# 500      80000      0.28     1778.28
# 500      160000      0.24     2114.74
# 500      320000      0.20     2514.87

# 1000      10000      0.56     1778.28
# 1000      20000      0.47     2114.74
# 1000      40000      0.40     2514.87
# 1000      80000      0.33     2990.70
# 1000      160000      0.28     3556.56
# 1000      320000      0.24     4229.49















