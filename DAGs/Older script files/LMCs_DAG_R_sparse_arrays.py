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
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from os.path import join, expanduser
import glob
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool, set_start_method
import tqdm
import networkx as nx
from collections import defaultdict, deque

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


######################### Generating a relations matrix using dense R ################################################

# @njit(parallel=True)
# def R_from_fc_numba(final_coords):
#     N=len(final_coords)
#     R = np.zeros((N,N)) # initialize the relations matrix
    
#     count = 0
#     for i in prange(N):
#         t_i=final_coords[i,0] #labeling the coordinates of point i. For some reason doing it this way instead of 
#         x_i=final_coords[i,1] #t_i, x_i, y_i, z_i = final_coords[i] is faster (19s vs 26s for N=10k). Also, 
#         y_i=final_coords[i,2] #it is important to define them at this stage rather than in the second loop for speed!
#         z_i=final_coords[i,3]
        
#         for j in range(i+1,N): #the start is i+1 because of natural labeling j<=1 cannot be to the future of i
#             t_j=final_coords[j,0] #labeling the coordinates of point j
#             x_j=final_coords[j,1] 
#             y_j=final_coords[j,2]
#             z_j=final_coords[j,3]
            
#                 # Check the condition
#             if (t_j - t_i)**2 > (x_j - x_i)**2 + (y_j - y_i)**2 + (z_j - z_i)**2: #this condition says 
#                 #that if the timelike separation between j and i is more than the spacelike separation
#                 #then j is to the future of i.
#                 #if t_j-t_i > np.linalg.norm(final_coords[j,1:]-final_coords[i,1:]): This condition is slower!
#                 R[i,j]=1
#                 count += 1
#             #   print("Found a relation!")
#     return R, count

# def R_sparse_from_fc(final_coords):
#     t0 = time.time()
#     R, count = R_from_fc_numba(final_coords)
#     t1 = time.time()
#     print(f"---Finding R took {t1 - t0: .2f} seconds---") 
    
#     R_sparse = csr_matrix(R)
#     t1 = time.time()
#     print(f"Converting to csr_matrix took: {t1 - t0: .2f} s")
    
#     return R_sparse


######################### Generating a relations matrix using arrays ################################################

@njit # For N = 5k time goes down from 5s --> 0.09s
def R_sparse_arrays(final_coords):
    N = len(final_coords)
    
    # Estimate the maximum number of relations (upper bound for memory preallocation)
    max_relations = int(N * N/2)  # Adjust based on expected sparsity
    
    #row_indices = np.empty(max_relations, dtype=np.int32)
    row_ptr = np.zeros(N + 1, dtype=np.int32) ##Changing the function to remove reduntant values in row_indices using CSR format for storing indices
    col_indices = np.empty(max_relations, dtype=np.int32)
      

    num_edges_E = 0  # Counter for actual relations found
    
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

def R_from_fc_arrays(final_coords):
    t0 = time.time()
    
    row_indices, col_indices, values, N, numofrel = R_sparse_arrays(final_coords)
    t1 = time.time()
    print(f"Finding the arrays for R_sparse took: {t1 - t0: .2f} s")
    
    R_sparse = csr_matrix((values, (row_indices, col_indices)), shape=(N, N))
    t2 = time.time()
    print(f"Constructing the csr_matrix took: {t2 - t1: .2f} s")
    
    return R_sparse

def csr_matrices_equal(x, y):
    if (np.all(x.indices == y.indices)
    and np.all(x.indptr == y.indptr)
    and np.allclose(x.data, y.data)):
        print("Matrices are identical!")
    else:
        print("Matrices differ!")
    return

def L_from_R_sparse(R_sparse):
    t00 = time.time()
    
    R_2 = R_sparse @ R_sparse
    t2 = time.time()
    print(f"Squaring R_sparse took: {t2 - t00: .2f} s")
    
    # Change all the data of R_2 to 1s
    R_2.data = np.ones_like(R_2.data) # if I want to use np.ones then I would first have to define the array size etc.
    
    # Secret formula
    L = R_sparse - R_2
    
    t1 = time.time()
    print(f"Total time to build L from R_sparse: {t1 - t00:.2f} s")
    
    return L

def L_from_DAG(R_sparse):
    t00 = time.time()
    
    # create G
    G = nx.from_scipy_sparse_array(csr_matrix(R_sparse), create_using=nx.DiGraph)
    t1 = time.time()
    print(f"Time taken to convert R_sparse to G: {t1 - t00:.2f} s")

    # transitive reduction
    G_reduced = nx.transitive_reduction(G)
    t2 = time.time()
    print(f"Time taken to transitively reduce R_sparse to G: {t2 - t1:.2f} s")

    # converting back to csr_matrix
    L = nx.to_scipy_sparse_array(G_reduced, format='csr')
    t3 = time.time()
    print(f"Time taken to convert G_reduced to L_sparse: {t3 - t2:.2f} s")
    
    print(f"Total time taken to build L from DAG: {t3 - t00:.2f} s")
    
    return L, G_reduced

def DAG_neg_from_R(R_sparse):
    t00 = time.time()
    
    # create G
    G = nx.from_scipy_sparse_array(csr_matrix(R_sparse), create_using=nx.DiGraph)
    t1 = time.time()
    print(f"Time taken to convert R_sparse to G: {t1 - t00:.2f} s")
    
    #create negative G
    G_neg = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        G_neg.add_edge(u, v, weight=-1)
    t2 = time.time()
    print(f"Time taken to generate G_neg: {t2 - t1:.2f} s")
    return G_neg

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

## Finding all the longest paths using all_simple_paths
def find_longest_paths_fail(G, source, target):
    all_paths = list(nx.all_simple_paths(G, source=source, target=target))
    max_len = max(len(path) for path in all_paths)
    longest_paths = [path for path in all_paths if len(path) == max_len]
    return longest_paths

## Finding LMCs using R to create G, then G_neg, and finally, all_shortest_paths
def LMCs_from_G_neg(G_neg, source, target):
    t0 = time.time()
    LMCs = list(nx.all_shortest_paths(G_neg, source, target, weight="weight", method="bellman-ford"))
    t1 = time.time()
    print(f"Time taken to find all the LMCs using a DiGraph: {t1 - t0:.2f} s")
    return LMCs

##################### Finding LMCs using edge relaxation
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
    
    # # Step 1: Build adjacency list
    # t0 = time.time()
    # adj_list = defaultdict(list)
    # edges = zip(row_indices, col_indices)
    # for u, v in edges:
    #     adj_list[u].append(v)
    t1 = time.time()
    # print(f"Time taken to construct adjacency list: {t1 - t0: .2f} s")
    
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
    def backtrack(path, node):
        if node == s:
            all_paths.append(path[::-1])  # Reverse path
            return
        for parent in pred[node]:
            backtrack(path + [parent], parent)
    
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
    
    # def get_longest_path_levels(pred, s, t):
    #     levels = {}  # Dictionary to store nodes at each level
    #     queue = deque([(t, 0)])  # Start from target t with level 0

    #     while queue:
    #         node, level = queue.popleft()

    #         # Add node to its corresponding level
    #         if level not in levels:
    #             levels[level] = set()
    #             levels[level].add(node)

    #         # Stop if we reached s
    #         if node == s:
    #             continue
        
    #     # Add predecessors to the queue with incremented level
    #         for parent in pred.get(node, []):
    #             queue.append((parent, level + 1))

    #     # Convert dictionary to a sorted list of lists
    #     sorted_levels = [sorted(levels[l]) for l in sorted(levels.keys())]
    
    #     return sorted_levels

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

@njit
def find_longest_paths_numba(n, row_ptr, col_indices, s, t, max_pred=10):
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
    distance = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    distance[s] = 0  # Set source distance to 0
    
    # Initialize predecessors: 2D array (n, max_pred) and count array
    pred = np.full((n, max_pred), -1, dtype=np.int32)  # -1 means no predecessor
    pred_count = np.zeros(n, dtype=np.int32)  # Number of predecessors per node
    
    # Relaxation step: Iterate over all nodes
    for u in range(n):
        if distance[u] != np.iinfo(np.int32).max:  # If node u is reachable
            # Process all neighbors of u using CSR format
            for idx in range(row_ptr[u], row_ptr[u + 1]):
                v = col_indices[idx]  # Neighbor node
                new_dist = distance[u] - 1  # Assume edge weight is -1 for longest path
                if distance[v] == np.iinfo(np.int32).max or new_dist < distance[v]:
                    # Found a better (longer) path
                    distance[v] = new_dist
                    pred[v, 0] = u  # Reset predecessors
                    pred_count[v] = 1
                elif new_dist == distance[v]:
                    # Found an equally long path
                    if pred_count[v] < max_pred:
                        pred[v, pred_count[v]] = u  # Add predecessor
                        pred_count[v] += 1
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

@njit
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
    visited = np.zeros(n, dtype=np.bool_)
    queue = [T]
    visited[T] = True
    while queue:
        current = queue.pop(0)
        for i in range(pred_count[current]):
            u = pred[current, i]
            if not visited[u]:
                visited[u] = True
                queue.append(u)
    return visited
#@njit
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

    # Backtracking function to find all longest paths
    # def backtrack(path, node):
    #      if node == s:
    #          all_paths.append(path[::-1])  # Reverse path to get s -> t order
    #          return
    #      for i in range(pred_count[node]):
    #          parent = pred[node, i]
    #          backtrack(path + [parent], parent)
             
    # def backtrack_2(level, all_levels):
    #     if level == 0:
    #         all_levels[level].append(t)
        
    #     for child in all_levels[level]:
    #         if child == s:
    #             all_levels[:] = all_levels[::-1]
    #             return True
            
    #     if len(all_levels) == level + 1:
    #         all_levels.append([])
        
    #     for child in all_levels[level]:
    #         for parent in pred[child]:
    #             if parent not in all_levels[level + 1]:
    #                 all_levels[level + 1].append(parent)   
        
    #     if backtrack_2(level + 1, all_levels):
    #         return True
    


    # all_paths = []
    # if distance[t] != np.iinfo(np.int32).max:  # If target is reachable
    #      backtrack([t], t)
         
    # all_levels = [[]]
    # backtrack_2(0, all_levels)
    return all_levels

####### comparing NetworkX and matrix power to find L ######

if __name__== "__main__":
    
    N = 150000;
    fc = Minkowski_cube(N, 4, 0)
    # R_sparse = R_from_fc_arrays(fc)
    
    # L_from_R = L_from_R_sparse(R_sparse)
    t0 = time.time()
    row_indices, col_indices = R_sparse_arrays(fc)
    print(f"Finding the arrays for R_sparse took: {time.time() - t0: .2f} s")
    
    # G_rel_neg = DAG_neg_from_arrays(row_indices, col_indices)
    # LMCs = LMCs_from_G_neg(G_rel_neg, 0, N-1)
    
    #all_levels_recur, all_levels_iter = find_longest_paths(N, row_indices, col_indices, 0, N-1)
    #all_levels_iter = find_longest_paths(N, row_indices, col_indices, 0, N-1)
    #distance, pred, all_levels_iter, all_paths = find_longest_paths(N, row_indices, col_indices, 0, N-1)
    #max_size = max(len(v) for v in pred.values())

    all_levels_numba = numpy_LMCs(N, row_indices, col_indices, 0, N-1, 60)










####################### Finding the length of the LMCs using powers of R #############

# def causet_tau_c_files(folder_path, R_sparse, i, j):
#     # R_sparse stored on RAM all the time.
#     code_start = time.time()
#     R_powerk_ij = 1
#     k = 0
#     N = R_sparse.shape[0]
    
#     # Generate a CSR sparse identity matrix
#     R_k = identity(N, format='csr') 
    
#     while R_powerk_ij > 0: 
#         # R_k in RAM as well at all times!
        
#         k += 1      #putting it before makes the last run be the kth run
#         # R_power=np.linalg.matrix_power(R,k)[c,a]
#         #very slow code for large N
        
#         #New attempt that is supposed to use already caluclated power to find the next one.
#         t2 = time.time()
        
#         R_k = R_sparse @ R_k
#         # I think over here I should do R_k.data = np.ones_like(R_k.data) to make all values 1.
        
#         t3 = time.time()
        
#         # Print when multiplication complete
#         print(f"R^{k} found and it took {t3 - t2: .2f} s")
        
#         # Store R_k to npz file:
#         file_name = f'R_{k}.npz'
#         file_path = join(folder_path, file_name)
#         save_npz(file_path, R_k)
        
#         #we want to know how many maximal chains are there between c and a:
#         numofmaxchains = R_powerk_ij
        
#         R_powerk_ij = R_k[i,j] #If the ca-th element of R^k=0 then the longest chain is k-1 long
#                                 #this is because the power of R is the number of links in the chain
#                                             #the fact that at k-th power R^k_pq = 0 means there are
#                                             #no chains with k links, so the cardinality of the chain is k-1.
        
#     t1 = time.time()
#     print(f"Finding tau_c took: {t1 - code_start: .2f} s")
#     print(f"There are {numofmaxchains} maximal chains of length {k-1} between element {i} and {j}")
    
#     return k-1


# def causet_tau_c_files_parallel(folder_path, R_sparse, i, j):
#     # R_sparse stored on RAM all the time.
#     code_start = time.time()
#     R_powerk_ij = 1
#     k = 0
#     N = R_sparse.shape[0]
    
#     # Generate a CSR sparse identity matrix
#     R_k = identity(N, format='csr') 
    
#     while R_powerk_ij > 0: 
#         # R_k in RAM as well at all times!
        
#         k += 1      #putting it before makes the last run be the kth run
#         # R_power=np.linalg.matrix_power(R,k)[c,a]
#         #very slow code for large N
        
#         #New attempt that is supposed to use already caluclated power to find the next one.
#         t2 = time.time()
        
#         R_k = R_sparse @ R_k
#         #R_k = parallel_multiply(R_sparse, R_k)
        
#         ###### I think over here I should do R_k.data = np.ones_like(R_k.data) to make all values 1.
#         R_k.data = np.ones_like(R_k.data)
        
#         t3 = time.time()
        
#         # Print when multiplication complete
#         print(f"R^{k} found and it took {t3 - t2: .2f} s")
        
#         # Store R_k to npz file:
#         file_name = f'R_{k}.npz'
        
        
#         file_path = join(folder_path, file_name)
#         save_npz(file_path, R_k)
        
#         #we want to know how many maximal chains are there between c and a:
#         #numofmaxchains = R_powerk_ij ## This will not work if I reset data to ones for each power
        
#         R_powerk_ij = R_k[i,j] #If the ca-th element of R^k=0 then the longest chain is k-1 long
#                                 #this is because the power of R is the number of links in the chain
#                                             #the fact that at k-th power R^k_pq = 0 means there are
#                                             #no chains with k links, so the cardinality of the chain is k-1.
        
#     t1 = time.time()
#     print(f"Finding tau_c took: {t1 - code_start: .2f} s")
#     #print(f"There are {numofmaxchains} maximal chains of length {k-1} between element {i} and {j}")
    
#     return k-1


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

# def LMCs_files(N = 10000):
#     #initialize other variables
#     D, l, i, j = 4, 0.2, 0, N - 1
    
#     t00 = time.time()
#     parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "Saving Rd multiprocessing", "temp")
#     # Create a unique folder for this process
#     folder_name = f"temp_{os.getpid()}"
#     folder_path = join(parent_path, folder_name)
#     os.makedirs(folder_path, exist_ok=True)
    
#     fc = Minkowski_cube(N, D, l)
#     R = R_from_fc_files(fc)
#     k_max = causet_tau_c_files(folder_path, R, i, j)
#     L = L_from_R_files(folder_path)
#     all_levels = LMCs_v32_files(folder_path, L, k_max, i ,j)
    
#     tf = time.time()
#     print(f"---Total time to find LMCs for N = {N}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")
    
#     # Remove the folder for this process
#     os.rmdir(folder_path)
    
#     return fc, all_levels


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
      
# visualizing_LMCs(100000)

















