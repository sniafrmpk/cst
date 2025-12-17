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

def delta_analyzer_all_points(N, height):
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
        results.append((height, N, num_of_levels, rs, ts, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length

    """
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "LMCs_data", "Intervals")
    
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
    sprinkling = []
    num_of_levels = 0
    # Step 1: Group sprinklings by number of levels
    sprinkling_groups = defaultdict(list)
    
    # Iterate over rows
    for row in df.itertuples(index=False):  # `index=False` avoids including index in tuple
        level, t, r = row  # Unpack the tuple
        if level == 0: 
            sprinkling = []
            #sprinkling.append((level, t, r)) # s and t are also added to the levels. s is at level 0 and t is at levels k_max (chain length)
            continue
        if t == height:
            #sprinkling.append((level, t, r))
            if sprinkling:  # Add the current chunk to the list
                sprinkling_groups[num_of_levels].append(sprinkling)
        else:
            sprinkling.append((level, t, r))
            num_of_levels = level
    
    # Step 2: Working with groups of sprinklings with the same path length.
    # A chunk is a sprinkling, so called because it is a 'chunk' of rows in the csv file.
    
    # the final array of results 
    results = []

    for num_of_levels, chunk_list in sprinkling_groups.items(): # path length, sprinklings with that length
        rs = []
        ts = []
        for chunk in chunk_list:
            for level, t, r in chunk:
                # r and t are stored for all the sprinklings with the same path length
                rs.append(r) 
                ts.append(t)
        
        # Store the relevant results for each group of sprinklings as a tuple in results array
        results.append((height, N, num_of_levels, rs, ts, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
      
    return results

def delta_analyzer_middle_layers(N, height, middle_thickness):
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
        (height, N, path_length, round(mean_of_sups, 4), round(std_sups, 4), round(mean_of_means, 4), round(std_error,4), group_level_means, group_level_se, group_level_means_t, group_level_se_t, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length

    delta_data : Array with one row
    Stores the following for all the sprinklings with the same H and N.    
    [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    Where the delta is the average of the deltas of varrying path sizes produced in sprinklings with the same
    N and H. 
    
    """
    
    parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "DAGs", "LMCs_data", "Intervals")
    
    # # Create a unique folder for this process
    #folder_name =  folder+f" batch - Height {height}"
    #folder_name = f"Intervals"
    folder_name = f"D{D} - Height {height}"
    folder_path = join(parent_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # # File name
    
    file_name = f"spherical_coordinates_D{D}_H{height}_N{int(N/1000)}k.csv"
    file_path = join(folder_path, file_name)
    df = pd.read_csv(file_path, usecols=[1, 2, 3])
    
    # How many middle layers to consider = middle_thickness*2+1
    #middle_thickness = 5
    
    if middle_thickness: middle = True
    else: middle = False
    
    
    # Step 0: Split the file into sprinklings by finding rows where r is 0
    sprinklings = []
    sprinkling = []
    
        # Iterate over rows. Does not store s and t.
    for row in df.itertuples(index=False):  # `index=False` avoids including index in tuple
        level, t, r = row  # Unpack the tuple
        if level == 0: 
            sprinkling = []
            continue
        if t == height:
            # Add the current chunk to the list
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
        supremums = [] # contains arrays of the [sup_r, sup_level, num_of_levels] for each sprinkling

        # array to store the average r for each level for all sprinklings of the same path length
        chunk_levels_means_list = []
        chunk_levels_means_list_t = []
        
    # Find mean for each level of each sprinkling to make mean for sprinkling. 
        for chunk in chunk_list:
            all_levels = [[] for _ in range(num_of_levels)] # stores the nodes at each level. Basically the all_levels array from the og code
            all_levels_t = [[] for _ in range(num_of_levels)]
            sup_r, sup_level = 0, 0
            for level, t, r in chunk:
                #print(f"level = {level}, len(all_levels) = {len(all_levels)}")
                #if len(all_levels) < level: all_levels.append([])
                level = int(level)
                r = float(r)
                all_levels[level - 1].append(r) # creates a lists of nodes at each level 
                all_levels_t[level - 1].append(t)
                
                # Find supremum
                if r > sup_r: 
                    sup_r = r
                    sup_level = level
            
            # 0A. 
            # Append supremum data for this sprinkling
            supremums.append((sup_r, sup_level))
           
            # 2A.
   ########## Step 1 of determining delta: list of mean delta for each level of the sprinkling
            sprinkling_means = [np.mean(rs) for rs in all_levels] # stores mean for each level of the sprinkling
            # array of means of all levels of all sprinklings of the same chain length
            chunk_levels_means_list.append(np.array(sprinkling_means))
            
            # 3A.
            sprinkling_means_t = [np.mean(ts) for ts in all_levels_t]
            chunk_levels_means_list_t.append(np.array(sprinkling_means_t))

            #### 1A.
   ########## Step 2 of determining delta:
            if not middle:
                # Compute mean and std error for each sprinkling ignorant of the level
                mean_r = np.mean(sprinkling_means)
                # se_r = np.std(all_r_values) # Don't know what to do with the std error for each sprinkling
                
            else:
                # Find the central three levels
                mid = num_of_levels // 2 # Floor-division. n // 2 for n even = n / 2, otherwise its n / 2 - 1 / 2
                mean_r = np.mean(sprinkling_means[mid-middle_thickness : mid+middle_thickness]) # slices the middle three values and finds their mean
    
   ########## Step 3 of determining delta:
            # Store delta for this sprinkling in the array chunk_means representing the array of means for all the sprinklings with the same longest chain length
            chunk_means.append(mean_r)
            
            
            # reset all_levels for this sprinkling
            all_levels = []
            all_levels_t = []
            
        # 0B. Calculate the mean of the supremums for that group of sprinklings
        mean_of_sups = np.mean(np.array(supremums), axis=0)
        std_sups = np.std(np.array(supremums), axis=0)
        
        # 1B. Calculate the mean of the means for that group of sprinklings

######### Step 4 of determining delta:
        mean_of_means = np.mean(chunk_means)
        std_error = np.std(chunk_means)
        
        # 2B. Converts to a 2D array of shape (sprinkling number, level number)
        group_level_means_list = np.array(chunk_levels_means_list)
        group_level_means = np.mean(group_level_means_list, axis=0)
        group_level_se = np.std(group_level_means_list, axis=0)
        
        # 3B.
        group_level_means_list_t = np.array(chunk_levels_means_list_t)
        group_level_means_t = np.mean(group_level_means_list_t, axis=0)
        group_level_se_t = np.std(group_level_means_list_t, axis=0)
        
        # Store the relevant results for each group of sprinklings as a tuple in results array
        path_length = num_of_levels - 1
        results.append((height, N, path_length, np.round(mean_of_sups, 4), np.round(std_sups, 4), round(mean_of_means, 4), round(std_error,4), group_level_means, group_level_se, group_level_means_t, group_level_se_t, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
        
######### Step 5 of determining delta:
        # An array that stores the deltas for each group of spriklings 
        deltas_middle_layers.append(mean_of_means) 
        #print(results)
   
######### Step 6 of determining delta:
    ## 3. average delta for all the sprinklings
    delta = np.mean(deltas_middle_layers)  
    
    # The array / list I want to be returned
    ##### For intervals only!
    # l_0 = (height / N) ** 0.25 # L = 1
    # L = 1
    # rho = N / (height * L**3)
    V = np.pi/24*height**4
    rho = N/V
    l_0 = rho**(-0.25)
    #delta_data = [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    delta_data = [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / height, 3), round(delta / l_0, 3), round(height / l_0, 3)]
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

def all_delta_data(middle_thickness, heights, N_values):
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
    #[1, 10, 100, 250, 500, 1000]
    for height in heights:
        # N values
        # if height == 1:
        #     N_values = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 200000, 300000, 400000]
        # else:
        # N_values = [10000, 20000, 40000, 80000, 160000, 320000]  # List of N values
        # N_values = np.array(N_values)
        
        ## This is the loop in which we gather all the results 
        # sequentially for each N by calling delta_analyzer_all_levels(N)
        for N in N_values:
            # Call the function delta_analyzer_middle_layers and store only the delta_data tuple by using [1]
            results_N_height, delta_data_N_height = delta_analyzer_middle_layers(N, height, middle_thickness)
            
            # Iteratively store all the delta_data tuples in all_delta_data array
            delta_data_all.append(delta_data_N_height)
            
            results_all.append(results_N_height)
            
    # create dfs
    df_delta_data = pd.DataFrame(delta_data_all, columns=[
        'height','N','rho','l0','delta', 'delta_over_H', 'delta_over_l0','height_over_l0'
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

def δbyH_vs_N(df_delta_data):
    """
    df_delta_data = pd.DataFrame(all_deltas, columns=[
        'height','N','rho','l0','delta', 'delta_over_H', 'delta_over_l0', 'height_over_l0'
        ])
    """
    x = np.array(df_delta_data['N'])
    height = np.array(df_delta_data['height'])
    delta = np.array(df_delta_data['delta'])
    y = delta/height
    
    
    fig1, ax = plt.subplots(figsize=(20,10), dpi=400)
    
    # # scatter plot with color mapped by height
    # scatter = ax.scatter(
    #     x, y, s=20,
    #     c=df_delta_data['height'],
    #     cmap='tab10'
    #     )
    # # color bar
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Height')
    
    # Curve 
    groups = df_delta_data.groupby('height')
    fit_results = {}

    for H, group in groups:
        # Extract values for this height
        N_values = group['N'].values
        delta_over_H = (group['delta'] / group['height']).values  # y = δ/H
        
        # scatter for this height group
        ax.scatter(N_values, delta_over_H, s=40, label=f'Height={H}')
        
        # log–log transform, log in python is ln
        X = np.log(N_values)
        Y = np.log(delta_over_H)

        # Linear fit: Y = slope*X + intercept, where slope = -beta and intercept = log A
        slope, intercept = np.polyfit(X, Y, 1)

        beta = -slope             # minus sign because slope is negative
        A = np.exp(intercept)     # prefactor

        fit_results[H] = {
            "beta": beta,
            "A": A
            }
        # --- plot the fit curve with A and beta in legend ---
        N_fit = np.linspace(min(N_values), max(N_values), 200)
        y_fit = A * N_fit**(-beta)
        ax.plot(
            N_fit, y_fit, '--',
            label=f'H={H}: A={A:.3f}, β={beta:.3f}'
        )
        
        # make a second plot of the fit itself
        fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=400)
        ax2.scatter(X, Y, label='log of data', color='red')
        ax2.plot(X, slope * X + intercept, label=f'Fit: $\\beta = {beta:.3f}$, $A = {A: .3f}$', color='blue')
          #plt.plot(X, np.exp(slope_exp * N_values + intercept_exp), label=f'Fit "Exp": $\\alpha = {alpha:.4f}$', color='orange')
        ax2.set_xlabel('log(N)')
        ax2.set_ylabel('Log(δ/H)')
        ax2.set_title('Power-law: δ = A N^(-β)')
          #plt.title('Power-law: δ = A N^(-β), Exp-decay: δ = A exp(-alpha*N)')
        ax2.legend()
        ax2.grid(True)
        fig2.tight_layout()
        parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis")
        file_name = f'Power-law fit height_{H}.png'
        file_path = join(parent_path, file_name)
        fig2.savefig(file_path)
        plt.close(fig2)
    # Anotate points
    for xi, yi, Ni, H, l0 in zip(
        x, y, df_delta_data['N'], df_delta_data['height'], df_delta_data['l0']
    ):
        ax.annotate(
            f"{yi:.3f}, {H/l0:.2f})",
            xy=(xi, yi),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            fontsize=9
        )
    
    ax.set_xlabel('N', fontsize=20)
    ax.set_ylabel('δ/H', fontsize=20)
    ax.set_title('δ/H vs. N with labels: (δ/H, H/l₀)', fontsize=20)
    ax.legend(fontsize=12, loc='best')

    plt.tight_layout()
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis")
    file_name = f"δbyH vs. N.png"
    file_path = join(parent_path, file_name)
    plt.savefig(file_path)
    plt.show()
    
    
    return

def δ_vs_N_fit(N_values, height = 1, min_freq = 4, mid_layers = False):
    #### Testing data fitting for delta vs. N ######
    
    #592368
    # Initialize delta_values array
    delta_values = []
    
    # Initialize the array that will store all the tuples of 
    # (height, N, path_length, round(mean_of_sups, 4), round(std_sups, 4), round(mean_of_means, 4), round(std_error,4), group_level_means, group_level_se, group_level_means_t, group_level_se_t, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
    # for each sprinkling group identified by the path length
    
    all_results = []

    ## This is the loop in which we gather all the results 
    # sequentially for each N by calling delta_analyzer_all_levels(N)

    for N in N_values:
        all_results.extend(delta_analyzer_middle_layers(N, height, mid_layers)[0])

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["Height", "N", "num_levels", "mean_sup", "std_sup", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])
    #print(df_results[["N", "mean_r", "num_sprinklings", "num_levels"]])
    
    for N in N_values:
         subset = df_results[df_results["N"] == N]
         total_sprinklings = np.sum(subset["num_sprinklings"].values)
         print(f"Total sprinklings for N = {N}: {total_sprinklings}")
         
         subset_filtered = subset[subset["num_sprinklings"] >= min_freq]
         deltas = subset_filtered["mean_r"].values
         #deltas = subset_filtered["mean_sup"].values
         delta_values.append(np.mean(deltas))
         
         #print(subset_filtered[["N", "mean_r", "num_sprinklings", "num_levels"]])
         # Mean r contains the means of all the sprinkling groups regardless of the path length
         # deltas = np.array([tup[0] for tup in subset_filtered["mean_sup"].values])
         # delta_values.append(np.mean(deltas))
    print(delta_values)
    # Step 3: Process multiple N values and generate the plot
    
    # Choose data set based on height and N values
    # height = 1000
    # most_frequent_sprinkling = True
    # min_freq = 4
    # Call the delt_vs_N function
    #delta_vs_N(height, most_frequent_sprinkling, min_freq)

    # Step 5: Finding relation between delta and N:
    # preferred: convert to numpy array (works for scalar height)
    delta_values = np.array(delta_values)
    Y = np.log(delta_values / height)
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

def δ_vs_level(N_values, height, mid_layers):
    all_results = []

    ## This is the loop in which we gather all the results 
    # sequentially for each N by calling delta_analyzer_all_levels(N)

    for N in N_values:
        all_results.extend(delta_analyzer_middle_layers(N, height, mid_layers)[0])

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["Height", "N", "num_levels", "mean_sup", "std_sup", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])
    
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
        y_max = 0.15
        for _, row in subset_filtered.iterrows():
            group_level_means = np.array(row["group_level_means"])  # Convert to NumPy array
            group_level_se = np.array(row["group_level_se"])  # Convert to NumPy array
            levels = np.arange(1, len(group_level_means) + 1)  # Levels (indices)
            
            # Create the plot for r vs level with error bars
            plt.figure(figsize=(20, 10), dpi=300)
            plt.errorbar(levels, group_level_means, yerr=group_level_se, fmt='o', capsize=5, label=f'freq={max_chunks}')
            #plt.errorbar(levels[0:20], group_level_means[0:20], yerr=group_level_se[0:20], fmt='o', capsize=5, label=f'N={N}')
            plt.ylim(y_min, y_max)
            #plt.margins(y=0.1)
            plt.xlabel('Level')
            plt.ylabel('Mean Delta')
            plt.title(f'Mean Deltas vs Levels for N = {N} and most common chain length {num_levels} with frequency {max_chunks}')
            plt.legend()
            plt.grid(True)
            
            # Create a unique folder for this process
            parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis", "Deltas vs Levels")
            folder = f"Intervals"
            folder_path = join(parent_path, folder)
            file_name = f"N{N}_#levels_{levels.max()}_freq_{max_chunks}.png"
            file_path = join(folder_path, file_name)
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(file_path)
            
            plt.show()
    return

def δ_vs_t(N_values, height, min_freq):
    all_results = []

    ## This is the loop in which we gather all the results 
    # sequentially for each N by calling delta_analyzer_all_levels(N)

    for N in N_values:
        all_results.extend(delta_analyzer_all_points(N, height))

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["Height", "N", "num_levels", "rs", "ts", "num_sprinklings"])
    
    # # ============================
    # # NEW PLOTS: r vs. t for all points of all sprinklings with the same 
    # #             path length.
    # # ============================
    # plt.figure(figsize=(18, 12), dpi=300)

    for N in N_values:
        subset = df_results[df_results["N"] == N]  # Filter rows for current N
        
        #Checking how many sprinklings each N has
        total_sprinklings = np.sum(subset["num_sprinklings"].values)
        print(f"Total sprinklings for N = {N}: {total_sprinklings}")
        
        subset_filtered = subset[subset["num_sprinklings"] >= min_freq]
        
        # fixed ranges
        x_min, x_max = 0, height
        y_min, y_max = 0, 0.50
        
        for _, row in subset_filtered.iterrows():
            rs = np.array(row["rs"])  # Convert to NumPy array
            ts = np.array(row["ts"])  # Convert to NumPy array
            num_levels = row["num_levels"]
            freq = row["num_sprinklings"]
            
            # Create the plot for r vs level with error bars
            plt.figure(figsize=(20, 10), dpi=600)
            plt.scatter(ts, rs, s=1, marker=".")
            
            # fixed axis + equal scale
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.axis("equal")
            
            # gridlines
            # Set y-ticks at twice the usual density
            yticks = np.linspace(y_min, y_max, 11)  # 11 ticks → spacing of 0.05 if range=0.5
            plt.yticks(yticks)

            # Keep x-ticks default
            plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.7)
            
            plt.xlabel('t')
            plt.ylabel('radial distance from t-axis')
            plt.title(f'For N = {N} -- r vs. t for all sprinklings with path length {num_levels+1} with frequency {freq}')
            
            # Create a unique folder for this process
            parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis", "rs vs t")
            folder = f"Intervals"
            folder_path = join(parent_path, folder)
            file_name = f"N{N}_#levels_{num_levels}_freq_{freq}.png"
            file_path = join(folder_path, file_name)
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(file_path)
            
            plt.close()
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
    
# To merge csv files
# df1 = pd.read_csv("spherical_coordinates_N_1.csv")
# df2 = pd.read_csv("spherical_coordinates_N.csv")

# merged = pd.concat([df1, df2], ignore_index=True)

# merged.to_csv("spherical_coordinates_N.csv", index=False)

if __name__ == "__main__":
    
    # for Height 1 only
    Ns = [654,  1309,  2618,  3927,  5236,  6545,  7854,  9163, 10472, 13090, 15708, 18326, 20944, 26180, 39270, 52360, 74048, 104720, 148096, 209440, 296193, 418880, 592368, 1184736, 1675470]
    # Ns = [1675470]
    #1675470
    # For Height 1 and 10
    #Ns = [  654,  1309,  2618,  3927,  5236,  6545,  7854,  9163, 10472, 13090, 15708, 18326, 20944]
    
    
    mid_thickness = False
    # df_delta_data = all_delta_data(mid_layers)
    # δ_vs_density(df_delta_data)
    
    height = 1
    min_freq = 1
    
    # δ_vs_t(Ns, height, mid_thickness)
    # δ_vs_level(height, mid_layers)
    # t_vs_level(height, mid_layers)
    δ_vs_N_fit(Ns, height, min_freq, mid_thickness)
    # # Read the files and create data frame
    #df_delta_data = all_delta_data(mid_thickness, heights, Ns)
    #print(df_delta_data)
    #δbyH_vs_N(df_delta_data)
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







