#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:28:11 2024

@author: naumanibrahim
"""
import numpy as np
import time
import random
from numba import njit
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

def Minkowski_cube(rho = 10000, D = 4, l=0):
    code_start_time=time.time()
   
    ###define things that we need###
    #N is the total number of points
    N=rho
    height = 1.0
    #coords is the unsorted array of coordinates of the points in the hypercube
    coords = np.zeros((N,D)) # this creates an array with N rows and D columns
    
    #Force coding the two points of interest, a and b, at (1,-l/2,0,0) and (1,l/2,0,0)
    coords[0,:]=(1,-l/2,0,0)
    #coords[1,:]=(1,l/2,0,0) # for now I want l = 0 and I don't want there to be to elements there
    #and also the coordinates for c, the comparison point to be at (0,0,0,0):
    coords[1,:]=(0,0,0,0)
    #coords[N-1,0] = height
    #trial_coord are the coordinates of one point. It will be generated repeatedly
    temp_coord = np.zeros((1,D)) #defining and initializing the temp coord as an array of
    #a single row with D columns filled with zeros. two braces are necessary in using np.zeros. 
    #np.zeros((1)) the same as np.zeros((1,1)).
    successes = 2 #defining the counter that tells when the coords array is full
    
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


######################### Generating csr arrays for edges ################################################
@njit # For N = 5k time goes down from 5s --> 0.09s
def Edges_csr_arrays(final_coords):
    N = len(final_coords)
    
    # Estimate the maximum number of relations (upper bound for memory preallocation)
    #num_edges_E = int(N * N/2)  # Adjust based on expected sparsity
    # First pass: count edges # I am compensating for memory by sacrificing time.
    num_edges_E = 0
    for i in range(N):
        t_i, x_i, y_i, z_i = final_coords[i]
        for j in range(i + 1, N):
            t_j, x_j, y_j, z_j = final_coords[j]
            if (t_j - t_i)**2 > (x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2:
                num_edges_E += 1
                
    #row_indices = np.empty(max_relations, dtype=np.int32)
    row_ptr = np.zeros(N + 1, dtype=np.uint64) ##Changing the function to remove reduntant values in row_indices using CSR format for storing indices
                                                #Chaning dtype here is crucial!
    col_indices = np.empty(num_edges_E, dtype=np.int32) #uint32 should also work. I checked and uint64 and uint32 give the same results for the same fc.
      
    num_edges_E = 0
    for i in range(N): # this cannot be prange because the value of count is not updated fast enough
                        # neither do I want to do something like row_indices.append because I am afraid that 
                        # the corresponding col indiex will not be stored at the same time and the index of some
                        # other parallel execution will add to the column array.
        t_i, x_i, y_i, z_i = final_coords[i]
        count_for_i = 0
        for j in range(i + 1, N):
            t_j, x_j, y_j, z_j = final_coords[j, 0], final_coords[j, 1], final_coords[j, 2], final_coords[j, 3]

            if (t_j - t_i)**2 > (x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2:
                #row_indices[count] = i
                col_indices[num_edges_E] = j
                num_edges_E += 1 # gives the same count as for the the dense array method! 
                               # but the number of non-zero entries is less when csr constructed
                               # The col_indices can never contain 0s because j > 0 in all loops.
                count_for_i += 1
        # update row_ptr where row_ptr[i] tells at which index in col_indices the outgoing edges of node i start
        # and row_ptr[i+1] tells where the next node's outgoing edges start in col_indices
        row_ptr[i+1] = row_ptr[i] + count_for_i # I was really confused here. I thought this should give the index for the last element of the 
                                                # outgoing list of edges in col_indices. The answer is that row_ptr[i] already stores the first element
                                                # of the list. So if there are 100 outgoing edges then and row_ptr[i] stores the first one we should add 99
                                                # to get the index of the last element for i, but since we add 100 we get the index for the first element of 
                                                # the list of outgoing edges for node i+1
    
   # values = np.ones(count, dtype=np.int8)   # All values will be 1                       
    # numofrel = count
    #return row_indices[:count], col_indices[:count], values[:count], N, numofrel # array[:n] = array[0:n], : is a slicing operator.
    return row_ptr, col_indices[:num_edges_E]



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

############ Finding LMCs using edge relaxation (Arrays + Numba) #######

@njit # Poduces: pred(~n x max_pred), pred_count and distance arrays
def find_longest_paths_numba(n, row_ptr, col_indices, s, t, max_pred=50):
    """
    Find all longest paths from source (s) to target (t) in a DAG using
    Bellman-Ford with negative weights, optimized with CSR format and arrays.
    
    Parameters:
    - n: Number of nodes
    - row_ptr: CSR row pointer array
    - col_indices: CSR column indices array
    - s: Source node
    - t: Target node
    - max_pred: Maximum number of predecessors per node (for array allocation)
    
    Returns:
    - List of longest paths from s to t
    """
    # Initialize distance array with a large integer value (infinity)
    distance = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    distance[s] = 0  # Set source distance to 0
    
    # Initialize predecessors: 2D array (n, max_pred) and count array
    pred = np.full((n, max_pred), -1, dtype=np.int64)  # -1 means no predecessor
    pred_count = np.zeros(n, dtype=np.int32)  # Number of predecessors per node
    
    # Relaxation step: Iterate over all nodes
    for u in range(n):
        if distance[u] != np.iinfo(np.int64).max:  # If node u is reachable
            # Process all neighbors of u using CSR format
            for idx in range(row_ptr[u], row_ptr[u + 1]):
                v = col_indices[idx]  # Neighbor node
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
   # Step 2: Iterative backtracking to find all longest paths
   #  all_paths = []
   #  if distance[t] != np.iinfo(np.int32).max:  # If target is reachable
   #      stack = [(t, [t])]  # Stack entries: (current_node, current_path)
   #      while stack:
   #          node, path = stack.pop()  # Get the current node and path
   #          if node == s:
   #              # Reached the source; add the path (reversed) to results
   #              all_paths.append(path[::-1])
   #          else:
   #              # Add all predecessors to the stack
   #              for i in range(pred_count[node]):
   #                  parent = pred[node, i]
   #                  stack.append((parent, path + [parent]))
   # # Backtracking function to find all longest paths
    # def backtrack(path, node):
    #     if node == s:
    #         all_paths.append(path[::-1])  # Reverse path to get s -> t order
    #         return
    #     for i in range(pred_count[node]):
    #         parent = pred[node, i]
    #         backtrack(path + [parent], parent)

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
    visited = np.zeros(n, dtype=np.bool_) # Zeros are interpreted as False!
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
        list: List of lists where index i contains nodes at distance i from S
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
def numpy_LMCs(n, row_ptr, col_indices, s, t, max_pred):
    t0 = time.time()
    # Find the pred, pred_count and distance arrays
    distance, pred, pred_count = find_longest_paths_numba(n, row_ptr, col_indices, s, t, max_pred)
    t1 = time.time()
    print(f"Time taken to make pred arrays using numba: {t1 - t0:.2f} s")
    
    # Find the nodes on the LMCs
    visited = find_nodes_on_longest_paths_numba(s, t, pred, pred_count, n)
    t2 = time.time()
    print(f"Time taken to make visited array using numba: {t2 - t1:.2f} s")

    # Find the all_levels array
    all_levels = find_nodes_by_level_numba(s, t, visited, distance, n)
    t3 = time.time()
    print(f"Time taken to make all_levels: {t3 - t2:.2f} s")

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
def Visualizing_LMCs_DAG(N):
    
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
    t00 = time.time()
    
    # Create fc
    fc = Minkowski_cube(N, D, l)
    
    # Create rows and columns for the edges
    t0 = time.time()
    row_indices, col_indices = Edges_csr_arrays(fc)
    print(f"Finding the arrays for the edges took: {time.time() - t0: .2f} s")
    print(f"The number of relations, E, for N = {N/1000}k is: E = {len(col_indices):.2e}")
    # Find all_levels
    all_levels = numpy_LMCs(N, row_indices, col_indices, i, j, 250)
    tf = time.time()
    print(f"---Total time to find LMCs for N = {N}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")
    
    # Save files
    # Because the code will be run in parallel I want each pair of fc and R to have a unique timestamp from day, down to microseconds #####
    # timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 100_000) % 100_000}"
    parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "DAGs", "LMCs_data")
    
    # # Create a unique folder for this process
    folder_name = "Second batch"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # # File name
    
    file_name = f"spherical_coordinates_N{int(N/1000)}k.csv"
    file_path = join(folder_path, file_name)
    
    # # Save with compression
    # with h5py.File(file_path_arrays, "w") as f:
    #     f.create_dataset("final_coordinates", data=fc, compression="gzip")
    #     f.create_dataset("row_ptr", data=row_indices, compression="gzip")
    #     f.create_dataset("col_indices", data=col_indices, compression="gzip", chunks=True)
        
    #     # Convert the lists to JSON strings
    #     serialized_levels = [json.dumps(level) for level in all_levels]
    #     # Store the serialized data as strings
    #     f.create_dataset("all_levels", data=serialized_levels, compression="gzip")    
    # # Remove all the tpz files from temp:
    # files = glob.glob(folder_path+'/*')
    # for f in files:
    #     os.remove(f)
    
    #####################################################
    
    # all_elements = []
    # for l, sub_dict in all_levels.items():
    #     all_elements.append(sub_dict['filtered_elements']) # returns a list of lists starting with the filtered elements in the top-most layer.
    # all_elements.reverse() # inverts the list ordering
    
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
            all_points.append({
                'element': element,
                'level':level,
                't':t,
                'r':r,
                'theta':theta,
                'phi':phi
                })  # does not create a list of lists of lists as I had hoped.
    #                                         # creates a list of the coordinates of all the points.
    # all_points = np.array(all_points) # converts a list of arrays to an array
    # ###### I would like to convert these coordinates to (t, r) of spherical coordinates and ignore the theta and phi coordinate.
    # # the r coordinate is given by sqrt(x**2 + y**2 + z**2).
    # # all_points_spherical = np.column_stack((all_points[:, 0], np.sqrt(all_points[:, 1]**2 + all_points[:, 2]**2 + all_points[:, 3]**2)))
    # t = all_points[:, 0]
    # r = np.sqrt(all_points[:, 1]**2 + all_points[:, 2]**2 + all_points[:, 3]**2)
    # x = all_points[:, 1]
    # y = all_points[:, 2]
    # z = all_points[:, 3]
    
    # Create DataFrame
    df = pd.DataFrame(all_points)
    
    # saving to csv format
    df.to_csv(file_path, mode='a', index=False, header=False)
    
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

if __name__ == "__main__":
    
#     #set_start_method('fork')
#     for N in [120000, 140000, 160000, 200000, 300000]:
#         if N == 300000: num_trials = 25
#         if N == 200000: num_trials = 50
#         else: num_trials = 100
    N = 400000   
    num_workers = 1
    num_trials = 20
    
    #create the array that will go as input in the pool.map
    args = np.full(num_trials, N)
    
    # call the vizualization in  parallel
    pool = Pool(processes=num_workers)
    list(tqdm.tqdm(pool.imap_unordered(Visualizing_LMCs_DAG, args), total=num_trials))
# import tracemalloc

# tracemalloc.start()

# Visualizing_LMCs_DAG(400000)

# current, peak = tracemalloc.get_traced_memory()
# print(f"Memory allocation peak: {peak/1024**2:.2f} MB")
# tracemalloc.stop()
















