#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:28:11 2024

@author: naumanibrahim
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict 
import os
from os.path import join, expanduser
import plotly.express as px
##### Use Mean of Means of Means for each level and each sprinkling and each group of sprinklings

def delta_analyzer_all_levels(N, height):
    """
    Parameters
    ----------
    N : int
        Number of points in the sprinkling.
    height : int
        The hieght of the sprinkled region.

    Returns
    -------
    results : List
        Stores the following for each group of sprinklings with the same N, H and k_max = path length    
        (N, num_of_levels, round(mean_of_means, 3), round(std_error,3), len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length.

    """
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "LMCs_data")
    
    # # Create a unique folder for this process
    #folder_name =  folder+f" batch - Height {height}"
    folder_name = f"Height {height}"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # File name
    
    file_name = f"spherical_coordinates_N{int(N/1000)}k.csv"
    file_path = join(folder_path, file_name)
    
    # Read only the level, t and r from each row.
    df = pd.read_csv(file_path, usecols=[1, 2, 3]) 
    
    # Step 0: Split the file into sprinklings by finding rows where r is 0
    sprinklings = []
    sprinkling = []
    
    # Iterate over rows
    for row in df.itertuples(index=False):  # `index=False` avoids including index in tuple
        level, t, r = row  # Unpack the tuple
        if level == 0: 
            sprinkling = []
            continue
        if t == height:
             if sprinkling:  # Add the current chunk to the list
                 sprinklings.append(sprinkling)
        else:
            sprinkling.append((level, t, r))
    
    # Step 1: Group sprinklings by number of levels
    sprinkling_groups = defaultdict(list)

    for sprinkling in sprinklings:
        num_of_levels = len(set(level for level, t, r in sprinkling)) #length of the set containing all level values for that sprinkling. Same as max of level
        sprinkling_groups[num_of_levels].append(sprinkling)
    
    
    # Step 2: Compute mean of r and std error for each group sprinklings with unique num of chains
    # by using a Mean of means of means approach. This prevents levels that have many nodes to overpower the mean for the sprinkling
    # A chunk is a sprinkling, so called because it is a 'chunk' of rows in the csv file.
    results = []
    chunk_means = []

    for num_of_levels, chunk_list in sprinkling_groups.items(): # path length, sprinklings with that length
        chunk_means = []
        chunk_levels_means_list = []
    # Find mean for each level of each sprinkling to make mean for sprinkling. 
        for chunk in chunk_list:
            all_levels = [[] for _ in range(num_of_levels)] # stores the r for the points at each level. Basically the all_levels array from the og code
            
            for level, t, r in chunk:
                #print(f"level = {level}, len(all_levels) = {len(all_levels)}")
                #if len(all_levels) < level: all_levels.append([])
                
                # Only the r is stored
                all_levels[level-1].append(r) # creates a lists of nodes at each level 
            
            # list of mean delta for each level of the sprinkling
            sprinkling_means = [np.mean(rs) for rs in all_levels] # stores mean for each level of the sprinkling
            
            #### 1A. 
            # Compute mean and std error for each sprinkling ignorant of the level
            mean_r = np.mean(sprinkling_means)
    
            # se_r = np.std(all_r_values) # Don't know what to do with the std error for each sprinkling
            chunk_means.append(mean_r)
            
            #### 2A.
            # array of means of all levels of all sprinklings of the same chain length
            chunk_levels_means_list.append(np.array(sprinkling_means))
            
            # reset all_levels for this sprinkling
            all_levels = []
            
        
        # 1B. Calculate the mean of the means for that group of sprinklings
        mean_of_means = np.mean(chunk_means)
        std_error = np.std(chunk_means)
        
        # 2B.
        group_level_means_list = np.array(chunk_levels_means_list)
        group_level_means = np.mean(group_level_means_list, axis=0)
        group_level_se = np.std(group_level_means_list, axis=0)
        
        # Store the relevant results for each group of sprinklings as a tuple in results array
        results.append((height, N, num_of_levels, round(mean_of_means, 4), round(std_error,4), group_level_means, group_level_se, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
      
    return results

def delta_analyzer_middle_layers(N, height, middle):
    """

    Parameters
    ----------
    N : int
        Number of points in the sprinkling.
    height : int
        The hieght of the sprinkled region.
    middle : bool
        Tells if the delta for a sprinkling is calculated using the middle three layers.    
    
    Returns
    -------
    results : List
            Stores the following for each group of sprinklings with the same N, H and k_max = path length    
            (N, num_of_levels, round(mean_of_means, 3), round(std_error,3), len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length.

    delta_data : Array with one row
    Stores the following for all the sprinklings with the same H and N.    
    [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    Where the delta is the average of the deltas of varrying path sizes produced in sprinklings with the same
    N and H. 
    
    """
    
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "LMCs_data")
    
    # # Create a unique folder for this process
    #folder_name =  folder+f" batch - Height {height}"
    folder_name = f"Height {height}"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # # File name
    
    file_name = f"spherical_coordinates_N{int(N/1000)}k.csv"
    file_path = join(folder_path, file_name)
    df = pd.read_csv(file_path, usecols=[1, 2, 3])
    
    # Step 0: Split the file into sprinklings by finding rows where r is 0
    sprinklings = []
    sprinkling = []
    
        # Iterate over rows
    for row in df.itertuples(index=False):  # `index=False` avoids including index in tuple
        level, t, r = row  # Unpack the tuple
        if level == 0: 
            sprinkling = []
            continue
        if t == height:
             if sprinkling:  # Add the current chunk to the list
                 sprinklings.append(sprinkling)
        else:
            sprinkling.append((level, t, r))
    
    # Step 1: Group sprinklings by number of levels
    sprinkling_groups = defaultdict(list)

    for sprinkling in sprinklings:
        num_of_levels = len(set(level for level, t, r in sprinkling)) #length of the set containing all level values for that sprinkling. Same as max of level
        sprinkling_groups[num_of_levels].append(sprinkling)
    
    
    # Step 2: Compute mean of r and std error for each group sprinklings with unique num of chains
    # by using a Mean of means of means approach. This prevents levels that have many nodes to overpower the mean for the sprinkling
    
    # A chunk is a sprinkling, so called because it is a 'chunk' of rows in the csv file.
    
    results = []
    chunk_means = []
    
    # This variable is supposed to disregard the fact that sprinkling groups have varrying
    # path length. It just cares about height of the sprinkled region and the number of points.
    # For all sorinklings with the same height and N, it stores the mean of delta for the middle
    # three layers. Later on, we take a mean of these to get a represenative delta for all the 
    # sprinklings with a given height and N.
    deltas_middle_layers = []
    
    for num_of_levels, chunk_list in sprinkling_groups.items(): # path length, sprinklings with that length
        chunk_means = []
        
        # array to store the average r for each level for all sprinklings of the same path length
        chunk_levels_means_list = []
        chunk_levels_means_list_t = []
        
    # Find mean for each level of each sprinkling to make mean for sprinkling. 
        for chunk in chunk_list:
            all_levels = [[] for _ in range(num_of_levels)] # stores the nodes at each level. Basically the all_levels array from the og code
            all_levels_t = [[] for _ in range(num_of_levels)]
            
            for level, t, r in chunk:
                #print(f"level = {level}, len(all_levels) = {len(all_levels)}")
                #if len(all_levels) < level: all_levels.append([])
                all_levels[level - 1].append(r) # creates a lists of nodes at each level 
                all_levels_t[level - 1].append(t)
                
            # list of mean delta for each level of the sprinkling
            sprinkling_means = [np.mean(rs) for rs in all_levels] # stores mean for each level of the sprinkling
            sprinkling_means_t = [np.mean(ts) for ts in all_levels_t]
            
            #### 2A.
            # array of means of all levels of all sprinklings of the same chain length
            chunk_levels_means_list.append(np.array(sprinkling_means))
            chunk_levels_means_list_t.append(np.array(sprinkling_means_t))
            
            #### 1A.
            if not middle:
                # Compute mean and std error for each sprinkling ignorant of the level
                mean_r = np.mean(sprinkling_means)
                # se_r = np.std(all_r_values) # Don't know what to do with the std error for each sprinkling
                
            else:
                # Find the central three levels
                mid = num_of_levels // 2 # Floor-division. n // 2 for n even = n / 2, otherwise its n / 2 - 1 / 2
                mean_r = np.mean(sprinkling_means[mid-1 : mid+2]) # slices the middle three values and finds their mean
            
            # Store delta for this sprinkling in the array chunk_means representing the array of means for all the sprinklings with the same longest chain length
            chunk_means.append(mean_r)
            
            
            # reset all_levels for this sprinkling
            all_levels = []
            all_levels_t = []
            
        # 1B. Calculate the mean of the means for that group of sprinklings
        mean_of_means = np.mean(chunk_means)
        std_error = np.std(chunk_means)
        
        # 2B.
        # converts to a 2D array of shape (sprinkling number, level number)
        group_level_means_list = np.array(chunk_levels_means_list)
        group_level_means_list_t = np.array(chunk_levels_means_list_t)
        
        group_level_means = np.mean(group_level_means_list, axis=0)
        group_level_means_t = np.mean(group_level_means_list_t, axis=0)
        
        group_level_se = np.std(group_level_means_list, axis=0)
        group_level_se_t = np.std(group_level_means_list_t, axis=0)
        
        
        # Store the relevant results for each group of sprinklings as a tuple in results array
        results.append((height, N, num_of_levels, round(mean_of_means, 4), round(std_error,4), group_level_means, group_level_se, group_level_means_t, group_level_se_t, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
        
        # An array that stores the deltas for each group of spriklings 
        deltas_middle_layers.append(mean_of_means) 
        #print(results)
    
    ## 3. average delta for all the sprinklings
    delta = np.mean(deltas_middle_layers)  
    
    # The array / list I want to be returned
    l_0 = (height / N) ** 0.25 # L = 1
    L = 1
    rho = N / (height * L**3)
    delta_data = [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    
    return results, delta_data

def delta_vs_N(height=1, most_frequent_sprinkling=True, min_freq=4):
    """
    Plots delta (average tube thickness) vs. N for sprinkling regions of various heights

    Args:
        height: Height of the sprinkling region
        most_frequent_sprinkling = bool type, tells which type of plot to make
                                    using the most frequent chain size or any chain
                                    size that appeares min_freq times in the sprinklings
        min_freq = only gets activated if most_frequent_sprinkling is False

    Returns:
        Plots a delta vs. N either for the most frequent chain size in all the sprinklings of size N
        or the chain sizes that appear more than min_freq times in the group of sprinklings
        
    """
    
    N_values = [10000, 20000, 40000, 80000, 160000, 320000]  # List of N values
    # if height == 1:
    #     N_values = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 200000, 300000, 400000]
    N_values = np.array(N_values)

    # Initialize delta_values array
    delta_values = []
    
    # Initialize the array that will store all the tuples of 
    # (N, num_of_levels, round(mean_of_means, 3), round(std_error,3), len(chunk_list) 
    # for each sprinkling group identified by the chain length
    
    all_results = []

    ## This is the loop in which we gather all the results 
    # sequentially for each N by calling delta_analyzer_all_levels(N)

    for N in N_values:
        all_results.extend(delta_analyzer_all_levels(N, height))

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["Height", "N", "num_levels", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])

    # df_results looks so: # i.e. each set of sprinklings gets their own index
    #          N  num_levels  mean_r   SE_r  num_sprinklings
    # 0     5000           9   0.135  0.030          47
    # 1     5000          10   0.107  0.017          18
    # 2     5000           8   0.153  0.016          31
    # 3     5000           7   0.178  0.007           3
    # 4     5000          11   0.107  0.000           1
    # ..     ...         ...     ...    ...         ...
    # 69  300000          30   0.063  0.001           2
    # 70  300000          29   0.083  0.010          12
    # 71  300000          28   0.084  0.015           6
    # 72  300000          31   0.077  0.000           1
    # 73  300000          27   0.097  0.008           4

    # Step 4: Plot mean r vs N with error bars
    plt.figure(figsize=(18, 12), dpi=300)
    min_freq = 4

    for N in N_values:
        subset = df_results[df_results["N"] == N]
    
    #### Choose what kind of sprinklings to plot based on their frequency of appearance
    
    ## To plot groups with the most sprinklings
        # Find the maximum num_sprinklings for the current N
        if most_frequent_sprinkling:
            all_sprinklings_gt_min_freq = False
            off_set = 0.001
        
        # Which sprinkling type (based on the chain length it produces) is most frequent
            max_chunks = subset["num_sprinklings"].max()
        
        ## Filter the subset to only include rows where num_sprinklings is the maximum for this N
            subset_filtered = subset[subset["num_sprinklings"] == max_chunks]
          # For each N, subset_filtered looks like (and then resets for the next iteration of the loop):
          #       N  num_levels  mean_r  SE_r  num_sprinklings
          # 0  5000           9   0.135  0.03          47
        
        ## Append delta for this N in the delta_values array
            # subset_filtered["mean_r"] is a pandas Series with one row
            # and it produces 0    0.135 Name: mean_r, dtype: float64
            # to get the mean (0.135 in this case) use subset_filtered["mean_r"].values
            # which returns a numpy array [0.135] so to access the just the value
            # use values[0]
            delta_values.append(subset_filtered["mean_r"].values[0])
    ## Filter the subset to only include rows where num_sprinklings >= 10
        else: 
            all_sprinklings_gt_min_freq = True 
            off_set = 0.002
            subset_filtered = subset[subset["num_sprinklings"] >= min_freq]
            # subset["num_sprinklings"] >= min_freq returns a True/False list 
            # for each of the rows meeting this criterion.
            
            deltas = subset_filtered["mean_r"].values
            # values[0] would have given only the first of the list
            delta_values.append(np.mean(deltas))
    
    # Main command for plotting 
        plt.errorbar(
            subset_filtered["N"], subset_filtered["mean_r"], yerr=subset_filtered["SE_r"], fmt='o', capsize=5
            ) # Don't know what [N] does in [N] * len(subset_filtered)

    # Adding labels with the tuple (num_levels, num_sprinklings)
        for i, row in subset_filtered.iterrows():
            label = f"({int(row['num_levels'])}, {int(row['num_sprinklings'])})"
            plt.text(
                N+0.1, row["mean_r"]+off_set, label, 
                fontsize=9, ha='center', va='center', 
                color='black', alpha=0.7
                )
    plt.figtext(0.2, 0.90, "Tuples represent (chain length - 1, # of sprinklings with that length)", fontsize=10, ha='left', va='top', color='black')
       
    plt.xlabel("N")
    plt.ylabel("Mean delts")
    if most_frequent_sprinkling:
        title = f"Mean delta vs N - Height {height} - most frequent chain size"
        file_name = f"Most_freq_sprinkling_height_{height}.png"
    else:
        title = f"Mean delta vs N - Height {height} - freq(any chain size) >= {min_freq}"
        file_name = f"Sprinklings_gt_{min_freq}_height_{height}.png"
    plt.title(title, y=1.03)
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh's MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis", "By Height")
    file_path = join(parent_path, file_name)
    plt.savefig(file_path)
    plt.show()
    
   
    return

def all_delta_data(middle_layer):
    """
    Runs loops over values of height and N to collect all the lists
    delta_data = [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    with L = 1 and all values rounded to 3 dp, in a master list all_delta_data.
    
    Returns
    -------
    all_delta_data 

    """
    # Initialize the array that will store all the tuples of 
    # [height, N, rho, l_0, delta, delta / l_0, L/l_0, height / l_0]
    # for each ~100 sprinklings all with the same (N, height)
    delta_data_all = []
    results_all = []
    
    for height in [1, 10, 100, 250, 500, 1000]:
        # N values
        # if height == 1:
        #     N_values = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 200000, 300000, 400000]
        # else:
        N_values = [10000, 20000, 40000, 80000, 160000, 320000]  # List of N values
        N_values = np.array(N_values)
        
        ## This is the loop in which we gather all the results 
        # sequentially for each N by calling delta_analyzer_all_levels(N)
        for N in N_values:
            # Call the function delta_analyzer_middle_layers and store only the delta_data tuple by using [1]
            results_N_height, delta_data_N_height = delta_analyzer_middle_layers(N, height, middle_layer)
            
            # Iteratively store all the delta_data tuples in all_delta_data array
            delta_data_all.append(delta_data_N_height)
            
            results_all.append(results_all)
            
    # create dfs
    df_delta_data = pd.DataFrame(delta_data_all, columns=[
        'height','N','rho','l0','delta','delta_over_l0','L_over_l0','height_over_l0'
        ])
    
    # df_results = pd.DataFrame(results_all, columns=["Height", "N", "num_levels", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])


    #return df_delta_data, df_results
    return df_delta_data

def δ_vs_sqrtheight(df_delta_data):
    """
    df_delta_data = pd.DataFrame(all_deltas, columns=[
        'height','N','rho','l0','delta','delta_over_l0','L_over_l0','height_over_l0'
        ])
    """
    Height = df_delta_data['height']
    x = np.sqrt(Height)
    y = df_delta_data['delta']
    
    
    fig1, ax = plt.subplots(figsize=(18,12), dpi=400)
    ax.scatter(x, y, s=20)
    for xi, yi, Ni, H, l0 in zip(
        x, y, df_delta_data['N'], df_delta_data['height'], df_delta_data['l0']
    ):
        ax.annotate(
            f"({int(Ni/1000)}k, {H}, {l0:.2f})",
            xy=(xi, yi),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            fontsize=6
        )
    
    ax.set_xlabel('sqrt(height)', fontsize=10)
    ax.set_ylabel('δ', fontsize=10)
    ax.set_title('δ vs. sqrt(height) with (N, H, l₀) labels', fontsize=10)
    plt.tight_layout()
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis")
    file_name = f"δ vs. sqrt(height).png"
    file_path = join(parent_path, file_name)
    plt.savefig(file_path)
    plt.show()
    return

def δbyl0_vs_heightbyl0(df_delta_data):

    
    # # Plot an interactive 3D plot of δ/l₀ vs. log(height) and N
    # fig = px.scatter_3d(df, x='height', y='N', z='delta_over_l0',
    #                 color='L_over_l0',  # optional 4th variable
    #                 title='δ/l₀ vs. height and N')
    # fig.update_layout(
    #     title='δ/l₀ vs. log(height) & N',
    #     scene=dict(
    #         xaxis_title='log(height)',
    #         yaxis_title='N',
    #         zaxis_title='δ / l₀',
    #         xaxis=dict(type='log')
    #         )
    #     )
    # fig.write_html('delta_vs_height_N.html', auto_open=True)
    
    # # Plot an interactive 3D plot of δ/l₀ vs. height/l₀ and L=1/l₀
    # fig2 = px.scatter_3d(df, x='height_over_l0', y='L_over_l0', z='delta_over_l0',
    #                 color='rho',  # optional 4th variable
    #                 title='δ/l₀ vs. height and L=1 over l₀')
    # fig2.update_layout(
    #     title='δ/l₀ vs. vs. height and L=1 over l₀',
    #     scene=dict(
    #         xaxis_title='height / l₀',
    #         yaxis_title='L=1 / l₀',
    #         zaxis_title='δ / l₀',
    #         #xaxis=dict(type='log')
    #         )
    #     )
    # fig2.write_html('delta_vs_height_L_over_l0.html', auto_open=True)
  
    
    
    x = df_delta_data['height_over_l0']
    y = df_delta_data['delta_over_l0']
    
    fig2, ax = plt.subplots(figsize=(18,12), dpi=400)
    ax.scatter(x, y, s=20)
    for xi, yi, Ni, rhoi, Li in zip(
        x, y, df_delta_data['N'], df_delta_data['rho'], df_delta_data['L_over_l0']
    ):
        ax.annotate(
            f"({int(Ni/1000)}k, {rhoi}, {int(Ni/rhoi)},{Li:.1f})",
            xy=(xi, yi),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            fontsize=6
        )
    
    ax.set_xlabel('height / l₀', fontsize=10)
    ax.set_ylabel('δ / l₀', fontsize=10)
    ax.set_title('δ / l₀ vs. height / l₀ with (N, ρ, V, L/l₀) labels', fontsize=10)
    plt.tight_layout()
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro","GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis")
    file_name = f"δbyl₀ vs. heightbyl₀.png"
    file_path = join(parent_path, file_name)
    plt.savefig(file_path)
    plt.show()
    return

def δbyl0_vs_sqrtheightbyl0(df_delta_data):
    
    Height = df_delta_data['height_over_l0']
    Delta = df_delta_data['delta_over_l0']
    
    X = np.sqrt(Height)
    
    fig3, ax = plt.subplots(figsize=(18,12), dpi=400)
    ax.scatter(X, Delta, s=20)
    
    for xi, yi, Ni, H in zip(
        X, Delta, df_delta_data['N'], df_delta_data['height']
    ):
        ax.annotate(
            f"({int(Ni/1000)}k, {H})",
            xy=(xi, yi),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            fontsize=6
        )
    ax.set_xlabel('sqrt(height/l0)', fontsize=10)
    ax.set_ylabel('δ/l0', fontsize=10)
    ax.set_title('δ/l0 vs. sqrt(height/l0) with (N, H) labels', fontsize=10)
    plt.tight_layout()
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro","GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis")
    file_name = f"δbyl0 vs. sqrt(heightbyl0).png"
    file_path = join(parent_path, file_name)
    plt.savefig(file_path)
    plt.show()
    return

def δ_vs_N_fit(height = 1, min_freq = 4, mid_layers = False):
    #### Testing data fitting for delta vs. N ######
    
    N_values = [10000, 20000, 40000, 80000, 160000, 320000] 
    
    # Initialize delta_values array
    delta_values = []
    
    # Initialize the array that will store all the tuples of 
    # (N, num_of_levels, round(mean_of_means, 3), round(std_error,3), len(chunk_list) 
    # for each sprinkling group identified by the chain length
    
    all_results = []

    ## This is the loop in which we gather all the results 
    # sequentially for each N by calling delta_analyzer_all_levels(N)

    for N in N_values:
        all_results.extend(delta_analyzer_middle_layers(N, height, mid_layers)[0])

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["Height", "N", "num_levels", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])

    for N in N_values:
         subset = df_results[df_results["N"] == N]
         subset_filtered = subset[subset["num_sprinklings"] >= min_freq]
         # Mean r contains the means of all the sprinkling groups
         deltas = subset_filtered["mean_r"].values
         delta_values.append(np.mean(deltas))
    
    # print(np.round(delta_values,4))
    
    # Step 3: Process multiple N values and generate the plot
    
    # Choose data set based on height and N values
    # height = 1000
    # most_frequent_sprinkling = True
    # min_freq = 4
    # Call the delt_vs_N function
    #delta_vs_N(height, most_frequent_sprinkling, min_freq)

    # Step 5: Finding relation between delta and N:
    Y = np.log(delta_values)
    X = np.log(N_values)
    slope, intercept = np.polyfit(X, Y, 1)
    beta = -slope
    A = np.exp(intercept)
    
    if mid_layers: print(f'Beta with mid_layers for height {height} = {np.round(beta,4)}')
    else: print(f'Beta with all layers for height {height} = {np.round(beta,4)}')
    

#     # Fit to exponential decay delta = A exp(-alpha*N): log_delta = m * N + c
#     slope_exp, intercept_exp = np.polyfit(N_values, Y, 1)
#     alpha = -slope
#     A_exp = np.exp(intercept_exp)

#     # Print results
#     # Power-law
#     print(f"For Power-law relation, Estimated beta: {beta:.4f}")
#     print(f"For Power-law relation, Estimated A: {A:.4f}")

#     # Exp decay
#     print(f"For Exp decay, Estimated alpha: {alpha:.4f}")
#     print(f"for Exp decay, Estimated A_exp: {A_exp:.4f}")

#     # Plot 
#     plt.figure(figsize=(6, 4), dpi=300)
#     plt.scatter(X, delta_values, label='Data', color='red')
#     plt.plot(X, np.exp(slope * X + intercept), label=f'Fit "Power": $\\beta = {beta:.4f}$', color='blue')
# #     #plt.plot(X, np.exp(slope_exp * N_values + intercept_exp), label=f'Fit "Exp": $\\alpha = {alpha:.4f}$', color='orange')
#     plt.xlabel('log(N)')
#     plt.ylabel('δ')
#     plt.title('Power-law: δ = A N^(-β)')
# #     #plt.title('Power-law: δ = A N^(-β), Exp-decay: δ = A exp(-alpha*N)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'Power-law fit height_{height}.png')
#     #plt.semilogy(N_values, delta_values)
#     plt.show()
    
    return

def δ_vs_level(height, mid_layers):
    N_values = [10000, 20000, 40000, 80000, 160000, 320000]
    all_results = []

    ## This is the loop in which we gather all the results 
    # sequentially for each N by calling delta_analyzer_all_levels(N)

    for N in N_values:
        all_results.extend(delta_analyzer_middle_layers(N, height, mid_layers)[0])

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["Height", "N", "num_levels", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])
    
    # # ============================
    # # NEW PLOT: group_level_means vs Level Index for each N
    # # ============================
    # plt.figure(figsize=(18, 12), dpi=300)

    for N in N_values:
        subset = df_results[df_results["N"] == N]  # Filter rows for current N
        ## Filtering based on freq
        # min_freq = 10
        # subset_filtered = subset[subset["num_chunks"] >= min_freq]
        
        ## Filtering the most common
        max_chunks = subset["num_sprinklings"].max()
        subset_filtered = subset[subset["num_sprinklings"] == max_chunks]
        
        num_levels = subset_filtered["num_levels"].iloc[0]

        y_min = 0
        y_max = 0.6
        for _, row in subset_filtered.iterrows():
            group_level_means = np.array(row["group_level_means"])  # Convert to NumPy array
            group_level_se = np.array(row["group_level_se"])  # Convert to NumPy array
            levels = np.arange(1, len(group_level_means) + 1)  # Levels (indices)
            
            # Create the plot for r vs level with error bars
            plt.figure(figsize=(20, 10), dpi=300)
            plt.errorbar(levels[0:20], group_level_means[0:20], yerr=group_level_se[0:20], fmt='o', capsize=5, label=f'N={N}')
            plt.ylim(y_min, y_max)
            #plt.margins(y=0.1)
            plt.xlabel('Level')
            plt.ylabel('Mean Delta')
            plt.title(f'Mean Deltas vs Levels for N = {N} and most common chain length {num_levels} with frequency {max_chunks})')
            plt.legend()
            plt.grid(True)
            
            # Create a unique folder for this process
            parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis", "Deltas vs Levels")
            folder = f"Height {height}"
            folder_path = join(parent_path, folder)
            file_name = f"H{height}_N{N}_#levels_{levels.max()}.png"
            file_path = join(folder_path, file_name)
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(file_path)
            
            plt.show()
    return

def t_vs_level(height, mid_layers):
    N_values = [10000, 20000, 40000, 80000, 160000, 320000]
    all_results = []

    ## This is the loop in which we gather all the results 
    # sequentially for each N by calling delta_analyzer_all_levels(N)

    for N in N_values:
        all_results.extend(delta_analyzer_middle_layers(N, height, mid_layers)[0])

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["Height", "N", "num_levels", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])
    
    # # ============================
    # # NEW PLOT: group_level_means_t vs Level Index for each N
    # # ============================
    # plt.figure(figsize=(18, 12), dpi=300)
    H = height
    
    for N in N_values:
        subset = df_results[df_results["N"] == N]  # Filter rows for current N
        ## Filtering based on freq
        # min_freq = 10
        # subset_filtered = subset[subset["num_chunks"] >= min_freq]
        
        ## Filtering the most common
        max_chunks = subset["num_sprinklings"].max()
        subset_filtered = subset[subset["num_sprinklings"] == max_chunks]
        
        num_levels = subset_filtered["num_levels"].iloc[0]
        lmax = num_levels
        
        # y_min = 0
        # y_max = 0.6
        for _, row in subset_filtered.iterrows():
            group_level_means = np.array(row["group_level_means_t"])  # Convert to NumPy array
            group_level_se = np.array(row["group_level_se_t"])  # Convert to NumPy array
            levels = np.arange(1, len(group_level_means) + 1)  # Levels (indices)
            
            levels_con = levels[len(levels)//2-5:len(levels)//2+5]
            group_level_means_con = group_level_means[len(levels)//2-5:len(levels)//2+5]
            group_level_se_con = group_level_se[len(levels)//2-5:len(levels)//2+5]
            
            # theoretical level times
            theory = (levels) * (H / lmax)
            theory_con = (levels_con) * (H / lmax) 
            
            # Create the plot for r vs level with error bars
            plt.figure(figsize=(20, 10), dpi=300)
            plt.errorbar(levels_con, group_level_means_con, yerr=group_level_se_con, fmt='o', capsize=5, label=f'Empirical mean t ± SE')
            plt.plot(levels_con, theory_con, '-',
                     label=f'Theory: level×(H/ℓₘₐₓ) = level×({H}/{lmax})')

            # plt.ylim(y_min, y_max)
            plt.margins(y=0.1)
            plt.xlabel('Level')
            plt.ylabel('Mean t')
            plt.title(f'Mean t vs Levels for N = {N} and most common chain length {num_levels} with frequency {max_chunks})')
            plt.legend()
            plt.grid(True)
            
            # Create a unique folder for this process
            parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis", "ts vs Levels")
            folder = f"Height {height}"
            folder_path = join(parent_path, folder)
            file_name = f"H{height}_N{N}_#levels_{levels.max()}.png"
            file_path = join(folder_path, file_name)
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(file_path)
            
            plt.show()
    return

def δ_vs_density(df_delta_data):
      """
      df_delta_data = pd.DataFrame(all_deltas, columns=[
          'height','N','rho','l0','delta','delta_over_l0','L_over_l0','height_over_l0'
          ])
      """
      rho = df_delta_data['rho']
      x = 1/np.sqrt(np.sqrt(rho))
      y = df_delta_data['delta']
      
      
      fig1, ax = plt.subplots(figsize=(18,12), dpi=400)
      ax.scatter(x, y, s=20)
      for xi, yi, Ni, H, l0 in zip(
          x, y, df_delta_data['N'], df_delta_data['height'], df_delta_data['l0']
      ):
          ax.annotate(
              f"({int(Ni/1000)}k, {H}, {l0:.2f})",
              xy=(xi, yi),
              xytext=(0, 5),
              textcoords="offset points",
              ha='center',
              fontsize=6
          )
      
      ax.set_xlabel('rho^-0.25', fontsize=10)
      ax.set_ylabel('δ', fontsize=10)
      ax.set_title('δ vs. rho^-0.25 with (N, H, l₀) labels', fontsize=10)
      plt.tight_layout()
      parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis")
      file_name = f"δ vs. rho^-0.25.png"
      file_path = join(parent_path, file_name)
      plt.savefig(file_path)
      plt.show()
      return
    
if __name__ == "__main__":
    
    mid_layers = False
    df_delta_data = all_delta_data(mid_layers)
    δ_vs_density(df_delta_data)
    
    # height = 100
    # min_freq = 4
    
    
    # t_vs_level(height, mid_layers)
   # δ_vs_N_fit(height, min_freq, mid_layers = False)
    # # Read the files and create data frame
    #all_deltas = all_delta_data()
    
    # df = pd.DataFrame(all_deltas, columns=[
    #     'height','N','rho','l0','delta','delta_over_l0','L_over_l0','height_over_l0'
    #     ])
    # # Plot 2 of δ vs. sqrt(height)
    # δ_vs_sqrtheight(df)
    
    # # Plot 3 of δ/l₀ vs. height/l₀ on a scatter plot
    # δbyl0_vs_heightbyl0(df)
    
    # # Plot 4 of δ/l_0 vs. \sqrt(height/l_0)
    # δbyl0_vs_sqrtheightbyl0(df)
    
    
    
    
# ####### Finding the slope of ln delta vs. ln N plot ######
#     Y = np.log(delta_values)
#     X = np.log(N_values)
#     #print(X)

#     # Fit to Power-law delta = A N^-\beta: log_delta = m * log_N + c
#     # np.polyfit(x, y, deg) fits a polynomial of degree deg to the data (x, y).

#     slope, intercept = np.polyfit(X, Y, 1)
#     beta = -slope
#     A = np.exp(intercept)

#     # Fit to exponential decay delta = A exp(-alpha*N): log_delta = m * N + c
#     slope_exp, intercept_exp = np.polyfit(N_values, Y, 1)
#     alpha = -slope
#     A_exp = np.exp(intercept_exp)

#     # Print results
#     # Power-law
#     print(f"For Power-law relation, Estimated beta: {beta:.4f}")
#     print(f"For Power-law relation, Estimated A: {A:.4f}")

#     # Exp decay
#     print(f"For Exp decay, Estimated alpha: {alpha:.4f}")
#     print(f"for Exp decay, Estimated A_exp: {A_exp:.4f}")

#     # Plot 
#     plt.figure(figsize=(6, 4), dpi=300)
#     plt.scatter(X, delta_values, label='Data', color='red')
#     plt.plot(X, np.exp(slope * X + intercept), label=f'Fit "Power": $\\beta = {beta:.4f}$', color='blue')
#     #plt.plot(X, np.exp(slope_exp * N_values + intercept_exp), label=f'Fit "Exp": $\\alpha = {alpha:.4f}$', color='orange')
#     plt.xlabel('log(N)')
#     plt.ylabel('δ')
#     plt.title('Power-law: δ = A N^(-β)')
#     #plt.title('Power-law: δ = A N^(-β), Exp-decay: δ = A exp(-alpha*N)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'Power-law fit height_{height}.png')
#     #plt.semilogy(N_values, delta_values)
#     plt.show()





 # Plot delta / l_0 vs. 
 # plt.figure(figsize=(18, 12), dpi=300)
 # plt.errorbar(
 #     all_delta_data[], subset_filtered["mean_r"], yerr=subset_filtered["SE_r"], fmt='o', capsize=5
 #     ) # Don't know what [N] does in [N] * len(subset_filtered)

 # # Adding labels with the tuple (num_levels, num_sprinklings)
 # for i, row in subset_filtered.iterrows():
 #     label = f"({int(row['num_levels'])}, {int(row['num_sprinklings'])})"
 #     plt.text(
 #         N+0.1, row["mean_r"]+off_set, label, 
 #         fontsize=9, ha='center', va='center', 
 #         color='black', alpha=0.7
 #         )
 # plt.figtext(0.2, 0.90, "Tuples represent (chain length - 1, # of sprinklings with that length)", fontsize=10, ha='left', va='top', color='black')

 # plt.xlabel("N")
 # plt.ylabel("Mean delts")
 # if most_frequent_sprinkling:
 #     title = f"Mean delta vs N - Height {height} - most frequent chain size"
 #     file_name = f"Most_freq_sprinkling_height_{height}.png"
 # else:
 #     title = f"Mean delta vs N - Height {height} - freq(any chain size) >= {min_freq}"
 #     file_name = f"Sprinklings_gt_{min_freq}_height_{height}.png"
 # plt.title(title, y=1.03)
 # parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis", "By Height")
 # file_path = join(parent_path, file_name)
 # plt.savefig(file_path)
 # plt.show()







