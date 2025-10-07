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
import scipy.sparse
from scipy.sparse import csr_matrix, lil_matrix, identity, coo_matrix
import itertools
import matplotlib.pyplot as plt

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

def R_from_fc(final_coords):
    t0 = time.time()
    R = R_from_fc_numba(final_coords)
    t1 = time.time()
    print(f"---Finding R took {t1 - t0: .2f} seconds---") 
    return R

def causet_tau_c2(R, i, j):
    code_start = time.time()
    R_powerk_ij = 1
    k = 0
    N = len(R)
    
    R_sparse = csr_matrix(R)
    t1 = time.time()
    print(f"Converting to csr_matrix took: {t1 - code_start: .2f} s")
    
    # Generate a CSR sparse identity matrix
    R_k = identity(N, format='csr') 
    
    # Array where all the powers of R are stored
    Rd = []
    
    while R_powerk_ij > 0: 
        k += 1      #putting it before makes the last run be the kth run
        # R_power=np.linalg.matrix_power(R,k)[c,a]
        #very slow code for large N
        
        #New attempt that is supposed to use already caluclated power to find the next one.
        t2 = time.time()
        R_k = R_sparse @ R_k
        t3 = time.time()
        
        # Print when multiplication complete
        print(f"R^{k} found and it took {t3 - t2: .2f} s")
        
        # Append to Rd
        Rd.append(R_k)
        
        # Store R^2 ... No need now. Already stored in Rd
        #if k == 2:
        #    R_2 = R_k
        #Or I could use my function R_k_pq but then i would have to use a dictionary
        #i.e. memo that is accessible and updated for each iteration of the while loop.
        
        #we want to know how many maximal chains are there between c and a:
        numofmaxchains = R_powerk_ij
        
        R_powerk_ij = R_k[i,j] #If the ca-th element of R^k=0 then the longest chain is k-1 long
                                #this is because the power of R is the number of links in the chain
                                            #the fact that at k-th power R^k_pq = 0 means there are
                                            #no chains with k links, so the cardinality of the chain is k-1.
        
    t1 = time.time()
    print(f"Finding tau_c took: {t1 - code_start: .2f} s")
    print(f"There are {numofmaxchains} maximal chains of length {k-1} between element {i} and {j}")
    return k-1, Rd 

########Using sparse matrices###################

# def L_from_R(R):
#     N=len(R)
#     R_sparse = csr_matrix(R)
#     R_2=R_sparse @ R_sparse
    
#     R_coo=R_sparse.tocoo()
#     #R_2_coo=R_2.tocoo()
    
#     L = lil_matrix((N,N))
    
#     for i, j in zip(R_coo.row,R_coo.col):
#         #for j in prange(i+1,N):
#             if R_2[i,j]==0:
#                 L[i,j]=1  
#     return L

######### Using numba #####################333

def L_from_R2(Rd):
    
    t00=time.time()
    #N=len(R) # Input R as a full matrix
                # If input as a csr_matrix then use N = R.shape[1]
    R_sparse = Rd[0]
    R_2 = Rd[1]
    
    N = R_sparse.shape[1]
    
    # R_sparse=csr_matrix(R)
    # t1 = time.time()
    # print(f"Converting R to csr_matrix took: {t1-t00:.2f} s")
    # R_2 = R_sparse @ R_sparse
    # t2=time.time()
    # print(f"The multiplication to find R^2 took: {t2 - t1:.2f} s")
   
    #t0 = time.time()
    cR = R_sparse.tocoo()
    cR_row, cR_col = cR.row, cR.col # these are arrays that store the Row/Column indices of non-zero elements
    
    #Now check if R_2 does not have these indices use vectorized form
    #create a Boolean array of size N
    mask = R_2[cR_row,cR_col].A1 == 0 # .A1 converts these enteries of R_2 into a numpy array.
                                        # == 0 checks if the entries are zero and returns True or False
    
    # Now we want to create the links matrix whose entrie are 1 for the indices above that have a True as an entry
    L_row = cR_row[mask] # cR_row[mask] adds to the array L_row as elements the indices of cR that satisfy our condition 
    L_col = cR_col[mask] # similarly for the column information that needs to go into L
    # the data for at each of these L_row, L_col pairs needs to be 1
    L_data = np.ones_like(L_row) # ones_like() is safer than ones I think because it ensures to create an array of the same type as its argument in shape (size) and data type
    
    #t1=time.time()
    #print(f"Building ingredients for L took: {t1-t0:.2f} s")

    #t0=time.time()
    L = coo_matrix((L_data, (L_row, L_col)), shape=(N, N)).tocsr()
    t1=time.time()
    #print(f"Making L a csr_matrix took: {t1-t0:.2f} s")
    
    print(f"Total time to build L: {t1-t00:.2f} s")
    return L

def L_from_R_optimized(Rd):
    t00 = time.time()
    
    R_sparse = Rd[0]
    R_2 = Rd[1]
    
    # Change all the data of R_2 to 1s
    R_2.data = np.ones_like(R_2.data) # if I want to use np.ones then I would first have to define the array size etc.
    
    L = R_sparse - R_2
    
    t1 = time.time()
    print(f"Total time to build L (optimized): {t1 - t00:.2f} s")
    return L



######################Finding LMCs########################33
def LMCs_v1(L, k_max, A, i, j):
    t00 = time.time()
    A = int(A)
    Ld = []
    current_power = L.copy()
    for _ in range(k_max):
        Ld.append(current_power)
        current_power = current_power @ L
    t1 = time.time()
    print(f"It takes {t1 - t00:.2f} s to find and store the powers of L")    
    print(len(Ld))
    ##############Update aoa###########
    # aoa is an array of arrays that is A long. Each array in it will be k_max
    # long and would contain the indices of the elements that make up the 
    # LMCs. I will start top down. So each array
    # in aoa would have the index of j as its 0th element.
    
    # Initializing aoa
    aoa = np.full((A, k_max), j, dtype=int) #all elements have value j
    
    #updating the k_max-2 index of each sub array, corresponding to the k_max-1 th element of the chain
    counter = 0
    L_topower_kmaxmin1 = Ld[k_max-2] # L^(k_max-1) matrix is stored at Ld[k_max-2]
    # print(f"Rows with non-zero elements in L^k_max")
    # iterate over all the columns of the i-th row of L^(k_max - 1)
    # Huge thanks to cgpt (see chat history for explaination):
    row_start = L_topower_kmaxmin1.indptr[i]
    row_end = L_topower_kmaxmin1.indptr[i + 1]
    columns = L_topower_kmaxmin1.indices[row_start:row_end]
    print(columns.size)
    for m in columns:
        if L[m,j] == 1:
            aoa[counter,k_max-2] = m
            counter += 1
    print(counter)        
    t2 = time.time()
    print(f"It took {t2 - t1: .2f} s to find the {k_max-1: .1f}th elements")
    return aoa

def LMCs_v2(L, k_max, A, i, j):
    t00 = time.time()
    A = int(A)
    Ld = []
    current_power = L.copy()
    for _ in range(k_max):
        Ld.append(current_power)
        current_power = current_power @ L
    t1 = time.time()
    print(f"It takes {t1 - t00:.2f} s to find and store the powers of L")    
    
    #####Finding the k_max-1th elements:############
    k_maxmin1th_elements = []

    L_topower_kmaxmin1 = Ld[k_max-2] # L^(k_max-1) matrix is stored at Ld[k_max-2]
    # iterate over all the columns of the i-th row of L^(k_max - 1)
    # Huge thanks to cgpt (see chat history for explaination):
    row_start = L_topower_kmaxmin1.indptr[i]
    row_end = L_topower_kmaxmin1.indptr[i + 1]
    columns = L_topower_kmaxmin1.indices[row_start:row_end]
    
    for m in columns:
        if L[m,j] == 1:
            k_maxmin1th_elements.append(m)
   
    #####Finding the k_max-2th elements:############
    k_maxmin2th_elements = []        
    L_topower_kmaxmin2 = Ld[k_max-3]
    row_start = L_topower_kmaxmin2.indptr[i]
    row_end = L_topower_kmaxmin2.indptr[i + 1]
    columns = L_topower_kmaxmin2.indices[row_start:row_end]
    
    #### Creating dictionary for storing the kmax-2 elements related to each
    #### kmax-1 element
    kmax_min2_elements_related_to_each_kmax_min1_element = {oneof_k_maxmin1: [] for oneof_k_maxmin1 in k_maxmin1th_elements}
    
    for candidate in columns:
        isrelated = False # For each candidate suppose it is not related
        
        for oneof_k_maxmin1 in k_maxmin1th_elements:
            if L[candidate, oneof_k_maxmin1] == 1:
                kmax_min2_elements_related_to_each_kmax_min1_element[oneof_k_maxmin1].append(candidate)
                ## I would like two things to happen:
                    # 1. I want candidate to be added to k_maxmin2th_elements
                    # if it is related to any of the preceding elements. But
                    # I only wnat it to happen once. Not everytime the above 
                    # condition is met.
                    # 2. I would like to form a list of k_maxmin2 elements
                    # linked to each k_maxmin1 element.
                if not isrelated:
                    k_maxmin2th_elements.append(candidate)
                    isrelated = True

    #####Finding the k_max-3rd elements:############

        
    t2 = time.time()
    print(f"It took {t2 - t1: .2f} s to find the {k_max-1: .1f}th elements")
    return k_maxmin1th_elements

####### LMCs_v3 is the final version #######
def LMCs_v32(Rd,L, k_max, i, j):
    ######################## Finding powers of L ###############################
    t00 = time.time()
    # Ld = [L]
    # current_power = L
    # for _ in range(1,k_max):
    #     current_power = current_power @ L
    #     Ld.append(current_power)
        
    # t1 = time.time()
    # print(f"It takes {t1 - t00:.2f} s to find and store the powers of L")

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
        relationships = {element: [] for element in current_elements
                         }
        
        Rtopower = Rd[next_level-1]
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

############# Putting together all the functions ##############
def LMCs2(N = 10000, D = 4, l = 0.2, i = 0, j = 9999):
    t00 = time.time()
    fc = Minkowski_cube(N, D, l)
    R = R_from_fc(fc)
    k_max, Rd = causet_tau_c2(R, i, j)
    L = L_from_R_optimized(Rd)
    all_levels = LMCs_v32(Rd, L, k_max, i ,j)
    
    tf = time.time()
    print(f"---Total time to find LMCs for N = {N}, D = {D}, l = {l}, i = {i} and j = {j}: {tf - t00: .2f} s---")
    return fc, k_max, Rd, L, all_levels


##### Visualizing the LMCs #########################3

def visualizing_LMCs(fc, all_levels):
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
        # 
        
    N = len(fc)
    
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
    ###### I would like to convert these coordinates to (t, r, z) of polar coordinates and ignore the phi coordinate.
    # the r coordinate is given by sqrt(x**2 + y**2).
    
    all_points_polar = np.column_stack((all_points[:, 0], np.sqrt(all_points[:, 1]**2 + all_points[:, 2]**2), all_points[:, 3]))
    
    ###### Plotting all_points_polar as a scatter plot using matplotlib.pyplot as plt
    
    # polar r vs time
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.scatter(all_points_polar[:,1], all_points_polar[:,0], c='blue', marker='o')

    # Add labels
    ax1.set_ylabel('Time')
    ax1.set_xlabel('Polar r')
    
    # Set plot title
    ax1.set_title(f'r vs t for N = {N}')
    
    # polar r vs z
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.scatter(all_points_polar[:,1], all_points_polar[:,2], c='blue', marker='o')

    # Add labels
    ax2.set_ylabel('z')
    ax2.set_xlabel('Polar r')
    
    # Set plot title
    ax2.set_title(f'r vs z for N = {N}')
    
    # 3d scatter plot
    #fig3 = plt.figure()
    #ax3 = fig3.add_subplot(111, projection = '3d')
    #ax3.scatter(all_points_polar[:,0], all_points_polar[:,1], all_points_polar[:,2], c='blue', marker='o')
    
    # Add labels
    #ax3.set_ylabel('z')
    #ax3.set_xlabel('Polar r')
    #ax3.set_zlabel('z')
    
    # Show the plots
    plt.show()

    
    return all_elements, all_points_polar 

























###### Earlier attempt to find R^k_pq myself ###########
def R_powerk_pq(R,k,p,q, memo = None):
    
    #first time check and creation of the memo dictionary
    if memo is None:
        memo = {}
    #check if value for R_k_pq already calculated
    if (k,p,q) in memo:
        return memo[(k,p,q)]
    
    N = len(R)
    if k==1: #defining the base case
        result = R[p,q] 
    else:
        result = 0
        for i in range(N):
            result += R[p,i]*R_powerk_pq(R,k-1,i,q,memo) #previously I was using R_k_pq in stead of result
            #This was causing problems because I was using the same name for the functiona and the variable
    
    #store result in memo
    memo[(k,p,q)] = result
    
    #although this code gives the correct answers and the memoization
    #makes it fast it is not as fast as np.linalg.matrix_power(R,k)[p,q]
    #at least till k=8
    #making the code faster:
    #However, I think I can make the code faster by using the natural labeling
    #of elements in R and thus changing the range in the loop above.
    
    return result