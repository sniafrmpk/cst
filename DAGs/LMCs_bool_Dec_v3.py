#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 125 11:28:11 2025

Changing edge relaxation from adj[u] as list --> in-line
"""
import numpy as np
import time
import random
from numba import njit, prange, set_num_threads, types
from numba.typed import List
#from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from os.path import join
from pathlib import Path
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

def Minkowski_cube(rho = 10000, height = 1, D = 4, l=0):
    N = rho
    width = 1
    coords = np.zeros((N,D))

    # Fix reference point a at (height, -l/2, 0, ...) and origin c at index 1.
    # After sorting by time, c gets index 0 and a gets index N-1.
    coords[0,:] = [height,-l/2,0,0]
    coords[1,:] = [0,0,0,0]
    temp_coord = np.zeros((1,D))
    successes = 2

    while successes < N:
        temp_coord[0,0] = height * random.random()
        temp_coord[0,1:] = width * np.random.rand(1,D-1) - width * 0.5
        coords[successes,:] = temp_coord[0,:]
        successes += 1

    # Sort by time coordinate to impose natural labeling.
    final_coords = coords[np.argsort(coords[:,0])]
    return final_coords

######################## DiGraph with negative weights for reference ###################
def iter_weighted_edges_from_neighbors(Rp, N_total, weight=-1):
    for u in range(N_total):
        nbrs = get_row_neighbors(Rp, N_total, u)
        for v in nbrs:
            yield (u, v, weight)

def graph_from_neighbors_weighted(Rp, N_total, weight=-1):
    t0 = time.time()
    G = nx.DiGraph()
    G.add_weighted_edges_from(iter_weighted_edges_from_neighbors(Rp, N_total, weight))
    print(f"Created weighted DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges "
          f"in {time.time() - t0:.2f}s")
    return G

####################### Finding LMCs using NetworkX ################

## To get a networkx digraph
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

############# Creating R_packed_bool_array using Numba ############
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
    return (tb + 7) // 8 # not the byte index, but the total number of bytes needed. Minimum N = 2 gives tb = 1, so 1 byte needed (+7 ensures > 0 bytes < 2 bytes).

# --------- allocate ---------

@njit
def make_empty_packed_upper(N_total):
    """Allocate 1D packed upper-triangle bit array (uint8)."""
    return np.zeros(total_bytes_upper(N_total), dtype=np.uint8)

# --------- set / clear / get single (i,j) with i<j ---------*
@njit
def set_edge(P, N, i, j):
    if not (0 <= i < j < N):
        raise IndexError
    idx = bit_index(i, j, N)              # global bit index
    
    # 1. Which byte should j go to in row i 
    b   = idx >> 3              # byte index. 
                                # same as idx // 8, i.e., quotient without decimal. 
                                # >> is a bitwise operator that removes bits from the end
                                # x >> y is equivalent to x / 2**y
    
    # 2. Which bit should it go to
    k   = idx & 7               # bit position 0...7 (LSB-first)
                                # idx & 7 == idx mod 8 = idx % 8, gives the remainder when dividing by 8. MSB (most significant bit) - first within each byte
    
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
def get_row_neighbors(P, N_total, i):
    L = N_total - 1 - i
    out = List.empty_list(types.int64)
    if L <= 0:
        return out

    r0   = start_of_row(i, N_total)
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
                if j < N_total:      # redundant but harmless safety
                    out.append(j)
    return out

# Set number of threads to be used by numba. Might be important for running on cluster
# set_num_threads(8)

@njit(parallel=True)  # Works for any spacetime dimension >= 2
def R_packed(final_coords):
    """
    Build the packed upper-triangular relations array R for a Minkowski
    spacetime of arbitrary dimension.

    final_coords has shape (N, dim), with:
        - N = interval_size + 2
        - time coordinate at index 0
        - spatial coordinates at indices 1..dim-1
    """
    N_total = final_coords.shape[0]
    dim = final_coords.shape[1]

    R_array = make_empty_packed_upper(N_total)
    num_edges_E = 0
    counts = np.zeros(N_total, dtype=np.int32)  # Number of edges from each node
    for i in prange(N_total):
        t_i = final_coords[i, 0]
        c = 0
        for j in range(i + 1, N_total):
            t_j = final_coords[j, 0]
            dt = t_j - t_i
            dt2 = dt * dt

            # Spatial distance squared in (dim-1) spatial dimensions
            dist2 = 0.0
            for k in range(1, dim):
                dx = final_coords[j, k] - final_coords[i, k]
                dist2 += dx * dx
                if dist2 > dt2:
                    break

            if dt2 > dist2:
                set_edge(R_array, N_total, i, j)
                c += 1
        counts[i] = c
    num_edges_E = counts.sum()
    return R_array, num_edges_E

############ Finding LMCs using edge relaxation (R_packed + Numba) #######

#### In-line #######
INF_I32 = np.int32(2**31 - 1)

@njit
def relax_from_row_inline(u, P, N_total, distance, pred, pred_count, max_pred):
    # If u is not reachable from s, nothing to do
    du = distance[u]
    if du == INF_I32:
        return

    # Number of entries in row u: (u, v) for v = u+1..N-1
    L = N_total - 1 - u
    if L <= 0:
        return

    # Bit-index range [r0, rEnd) in the packed array for this row
    r0   = start_of_row(u, N_total)
    rEnd = r0 + L          # exclusive

    # Byte indices covering this row
    byte_first = r0 >> 3               # r0 // 8
    byte_last  = (rEnd - 1) >> 3       # (rEnd - 1) // 8

    # Bit positions (0..7, MSB-first) of first and last relevant bits
    first_bit = r0 & 7
    last_bit  = (rEnd - 1) & 7

    # Per-row invariants / local aliases
    new_dist = du - 1
    dist = distance
    pr   = pred
    pc   = pred_count

    for b in range(byte_first, byte_last + 1):
        byte_val = P[b]
        if byte_val == 0:
            continue  # no edges in this byte

        # Decide which bit positions in this byte belong to row u
        if b == byte_first:
            bit_start = first_bit
        else:
            bit_start = 0

        if b == byte_last:
            bit_stop = last_bit + 1   # exclusive
        else:
            bit_stop = 8

        base_bit_index = b << 3      # 8 * b

        # Scan only the relevant bits in this byte
        for bit in range(bit_start, bit_stop):
            # MSB-first bit test: bit 0 => 0x80, bit 1 => 0x40, ...
            if (byte_val & (0x80 >> bit)) == 0:
                continue

            g = base_bit_index + bit      # global bit index within packed array
            offset = g - r0               # 0..L-1 within this row
            v = u + 1 + offset            # column index (neighbor) in 0..N-1

            old = dist[v]

            if old == INF_I32 or new_dist < old:
                dist[v] = new_dist
                pr[v, 0] = u
                pc[v] = 1
            elif new_dist == old:
                c = pc[v]
                if c < max_pred:
                    pr[v, c] = u
                    pc[v] = c + 1
                else:
                    # keep the safety check
                    raise ValueError("max_pred not big enough!")
@njit
def find_longest_paths_numba_R_packed_inline(N_total, P, s, max_pred=50): # Checked with digraph code and it produces the same levels
    distance = np.full(N_total, INF_I32, dtype=np.int32)
    distance[s] = 0

    pred = np.full((N_total, max_pred), -1, dtype=np.int32)
    pred_count = np.zeros(N_total, dtype=np.int32)

    for u in range(N_total):
        relax_from_row_inline(u, P, N_total, distance, pred, pred_count, max_pred)

    return distance, pred, pred_count

#### adj[u] as list #####
@njit # Poduces: pred(~n x max_pred), pred_count and distance arrays
def find_longest_paths_numba_R_packed(n_total, R_packed, s, max_pred=50):
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
    distance = np.full(n_total, np.iinfo(np.int64).max, dtype=np.int64)
    distance[s] = 0  # Set source distance to 0
    
    # Initialize predecessors: 2D array (n_total, max_pred) and count array
    pred = np.full((n_total, max_pred), -1, dtype=np.int32)  # -1 means no predecessor
    pred_count = np.zeros(n_total, dtype=np.int32)  # Number of predecessors per node
    
    # Relaxation step: Iterate over all nodes
    for u in range(n_total):
        if distance[u] != np.iinfo(np.int64).max:  # If node u is reachable, i.e. is in the future of s

            #row_u = np.unpackbits(R_packed[u], bitorder='big')[:n_total]
            adj_u = get_row_neighbors(R_packed, n_total, u) # creates an int32 array containing indices of non_zero entries in row[u] of R_bool 
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


#### creating the levels ####
# 1. Visited array
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

# 2. Python function to form list of lists of elements by level
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
    levels = [[] for _ in range(max_dist + 1)] # There are max edges + 1 levels
    
    for u in range(n):
        if visited[u]:
            level = -distance[u]  # Convert negative distance to level
            levels[level].append(u)
    for sublist in levels: # sort in ascending order
        sublist.sort()
    return levels

#### calling function ####
# Python function to combine the previous results
def numpy_R_packed_LMCs(n, R_packed, s, t, max_pred, inline):
    t0 = time.time()
    # Find the pred, pred_count and distance arrays using inline calls to v in adj[u]
    if inline:
        distance, pred, pred_count = find_longest_paths_numba_R_packed_inline(n, R_packed, s, max_pred)
    # Find the pred, pred_count and distance arrays using adj[u] as list
    if not inline:
        distance, pred, pred_count = find_longest_paths_numba_R_packed(n, R_packed, s, max_pred)
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
def Visualizing_LMCs_R_packed_interval(interval_size, height, D, inline):
    
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
    N_total = interval_size + 2
    i, j = 0, N_total - 1
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
    all_levels, pred, pred_count = numpy_R_packed_LMCs(N_total, R, i, j, max_pred, inline)
    tf = time.time()
    #print(f"---Total time to find LMCs for N = {N_total}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")


    # ###### Now I want to associate to each element its coordinates using fc
    # all_levels is a list of list. to iterate over it:
    all_points = []
    dim = fc.shape[1]
    for level, elements in enumerate(all_levels):
        for element in elements:
            # Time coordinate
            t = fc[element, 0]

            # Up to three spatial coordinates; fall back to 0 if not present
            x = fc[element, 1] if dim > 1 else 0.0
            y = fc[element, 2] if dim > 2 else 0.0
            z = fc[element, 3] if dim > 3 else 0.0

            # find the preds of the element and store in list
            preds = pred[element, 0:pred_count[element]]

            # Dimension-aware radius / angles:
            # - D = 2 (1+1): r = |x|, only right/left, theta = 0/pi, phi = 0
            # - D = 3 (2+1): r in the (x,y) plane, theta = polar angle, phi = 0
            # - D >= 4 (3+1+): usual 3D spherical (x,y,z), ignoring extra spatial dims
            if D == 2:
                r = np.abs(x)
                phi = 0.0
                theta = 0.0 if x > 0 else np.pi  
            elif D == 3:
                r = np.sqrt(x**2 + y**2)
                if r == 0.0:
                    theta = 0.0
                    phi = 0.0
                else:
                    theta = np.arctan2(y, x)
                    phi = 0.0
            else:
                r = np.sqrt(x**2 + y**2 + z**2)
                if r == 0.0:
                    theta = 0.0
                    phi = 0.0
                else:
                    theta = np.arctan2(y, x)
                    phi = np.arccos(z / r)

            # all_points.append([element, level, t, r, theta, phi, preds, interval_size, height, D])
            all_points.append([element, level, t, r, theta, phi, preds])
    
    return all_points

def _helper(params):
    N_inside, height, D, inline = params
    return Visualizing_LMCs_R_packed_interval(N_inside, height, D, inline)
#def _init():
#     set_num_threads(32)
import csv

def mem(N_inside):
    return 256/3/(1184736**2)*N_inside**2
def recommended_workers(N_inside, mem_limit_gb):
    return mem_limit_gb / mem(N_inside)

if __name__ == "__main__":
    # Settings
    height = 10
    D = 4
    # inline is the modification to the code that does not generate adj[u] when running the graph algorithm
    inline = True

    parent_path = str(Path(__file__).parent / "LMCs_data" / "Intervals")
    folder_name = f"D {D} - Height {height}"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # inside_Ns = [100]
    # inside_Ns = [654,  1309,  2618,  3927,  5236,  6545,  7854,  9163, 10472, 13090, 15708, 18326, 20944, 26180, 39270]
    inside_Ns = [52360, 74048, 104720, 148096, 209440]
    # inside_Ns = [296193, 418880, 592368]
    # inside_Ns = [1184736, 1675470]
    num_trials = 120
    mem_limit_gb = 22  # memory limit in GB
    num_cores = os.cpu_count()

    program_start_time = time.perf_counter()
    for N_inside in inside_Ns:
        # Determine number of worker processes per N
        # for mac with memory limit of 22GB
        
        worker_count = max(1, min(num_cores, int(recommended_workers(N_inside, mem_limit_gb))))
        if N_inside >= 420000:
            num_trials = 90
        elif N_inside >= 290000:
            num_trials = 90

        file_name = f"spherical_coordinates_D{D}_H{height}_N{int(N_inside/1000)}k.csv"
        out_path = join(folder_path, file_name)

        # Open output file and write header. "a" ensure appending to file, not overwrite
        with open(out_path, "a", newline="") as fout:
            writer = csv.writer(fout)
          
            # write header only if file is empty
            if fout.tell() == 0:
                # writer.writerow(["element", "level", "t", "r", "theta", "phi", "preds", "Interval size", "Height", "D" ])
                writer.writerow(["element", "level", "t", "r", "theta", "phi", "pred"])
 

            # Spawn pool of workers
            #with Pool(processes=worker_count, initializer=_init) as pool:
            with Pool(processes=worker_count) as pool:
                # args: each worker gets the same N, repeated num_trials times
                args = [(N_inside, height, D, inline)] * num_trials
                start_time = time.perf_counter()
                counter = 0
                for rows in pool.imap_unordered(_helper, args):
                    writer.writerows(rows)
                    counter += 1
                    if counter % worker_count == 0: 
                        print(f"Finished {counter} sprinklings for N={int(N_inside/1000)}k in {time.perf_counter() - start_time:.2f}s")
        print(f"Finished writing CSV (using inline={inline} with further tweaks) for N={N_inside} with {worker_count} workers.")
    print(f"--- Total program time: {time.perf_counter() - program_start_time:.2f} s ---")
    # #   # testing numba vs regular python for making R
    # N = 25000
    # fc = Minkowski_interval(N, 1, 4) # produces fc with interval_size+2 coordinates
    
    # # Create rows and columns for the edges
    # t0 = time.perf_counter()
    
    # R, num_edges_E = R_packed(fc)
    # print(f'Time to make R for N = {N} without njit and prange: {time.perf_counter() - t0:.2f}s')










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










