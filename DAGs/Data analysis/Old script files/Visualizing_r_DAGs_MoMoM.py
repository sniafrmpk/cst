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

    for num_of_levels, chunk_list in sprinkling_groups.items(): # path length, sprinklings with that length
        chunk_means = []
        
    # Find mean for each level of each sprinkling to make mean for sprinkling. 
        for chunk in chunk_list:
            all_levels = [[] for _ in range(num_of_levels)] # stores the nodes at each level. Basically the all_levels array from the og code
            
            for level, t, r in chunk:
                #print(f"level = {level}, len(all_levels) = {len(all_levels)}")
                #if len(all_levels) < level: all_levels.append([])
                all_levels[level-1].append(r) # creates a lists of nodes at each level 
            
            sprinkling_means = [np.mean(rs) for rs in all_levels] # stores mean for each level of the sprinkling
            
            # Compute mean and std error for each sprinkling ignorant of the level
            mean_r = np.mean(sprinkling_means)
            
            # se_r = np.std(all_r_values) # Don't know what to do with the std error for each sprinkling
            chunk_means.append(mean_r)
            
            # reset all_levels for this sprinkling
            all_levels = []
            
        
        # Calculate the mean of the means for that group of sprinklings
        mean_of_means = np.mean(chunk_means)
        std_error = np.std(chunk_means)
        
        # Store the relevant results for each group of sprinklings as a tuple in results array
        results.append((N, num_of_levels, round(mean_of_means, 3), round(std_error,3), len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
        #print(results)
    return results

def delta_analyzer_middle_layers(N, height, middle):
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
    
    # This variable is supposed to disregard the fact that sprinkling groups are have varrying
    # path length. It just cares about height of the sprinkled region and the number of points.
    # For all sorinklings with the same height and N, it stores the mean of delta for the middle
    # three layers. Later on, we take a mean of these to get a represenative delta for all the 
    # sprinklings with a given height and N.
    deltas_middle_layers = []
    
    for num_of_levels, chunk_list in sprinkling_groups.items(): # path length, sprinklings with that length
        chunk_means = []
        
    # Find mean for each level of each sprinkling to make mean for sprinkling. 
        for chunk in chunk_list:
            all_levels = [[] for _ in range(num_of_levels)] # stores the nodes at each level. Basically the all_levels array from the og code
            
            for level, t, r in chunk:
                #print(f"level = {level}, len(all_levels) = {len(all_levels)}")
                #if len(all_levels) < level: all_levels.append([])
                all_levels[level - 1].append(r) # creates a lists of nodes at each level 
            
            sprinkling_means = [np.mean(rs) for rs in all_levels] # stores mean for each level of the sprinkling
            
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
            
        
        # Calculate the mean of the means for that group of sprinklings
        mean_of_means = np.mean(chunk_means)
        std_error = np.std(chunk_means)
        
        # Store the relevant results for each group of sprinklings as a tuple in results array
        results.append((N, num_of_levels, round(mean_of_means, 3), round(std_error,3), len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
        
        # An array that stores the deltas for each group of spriklings 
        deltas_middle_layers.append(mean_of_means) 
        #print(results)
    
    # average delta for all the sprinklings
    delta = np.mean(deltas_middle_layers)  
    
    # The array / list I want to be returned
    l_0 = (height / N) ** 0.25 # L = 1
    L = 1
    rho = N / (height*L**3)
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
    #folder = f"Height {height}"
    
    N_values = [10000, 20000, 40000, 80000, 160000, 320000]  # List of N values
    if height == 1:
        N_values = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 200000, 300000, 400000]

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
    df_results = pd.DataFrame(all_results, columns=["N", "num_levels", "mean_r", "SE_r", "num_sprinklings"])

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

    for N in N_values:
        subset = df_results[df_results["N"] == N]
    
    #### Choose what kind of sprinklings to plot based on their frequency of appearance
    
    ## To plot groups with the most sprinklings
        # Find the maximum num_sprinklings for the current N
        if most_frequent_sprinkling:
            all_sprinklings_gt_10 = False
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
    # Filter the subset to only include rows where num_sprinklings >= 10
        else: 
            all_sprinklings_gt_10 = True 
            off_set = 0.002
            #min_freq = 4
            subset_filtered = subset[subset["num_sprinklings"] >= min_freq]
    
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
    parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Data analysis", "By Height")
    file_path = join(parent_path, file_name)
    plt.savefig(file_path)
    plt.show()
    
    return

def all_delta_data():
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
            delta_data_N_height = delta_analyzer_middle_layers(N, height, True)[1]
            
            # Iteratively store all the delta_data tuples in all_delta_data array
            delta_data_all.append(delta_data_N_height)

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

    return delta_data_all

def δ_vs_sqrtheight(df):
    """
    df = pd.DataFrame(all_deltas, columns=[
        'height','N','rho','l0','delta','delta_over_l0','L_over_l0','height_over_l0'
        ])
    """
    Height = df['height']
    x = np.sqrt(Height)
    y = df['delta']
    
    
    fig1, ax = plt.subplots(figsize=(18,12), dpi=400)
    ax.scatter(x, y, s=20)
    for xi, yi, Ni, H, l0 in zip(
        x, y, df['N'], df['height'], df['l0']
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
    parent_path = join(expanduser("~"), "Desktop", "Desktop - Sheikh’s MacBook Pro", "GitRepos", "cst_longest_maximal_chains", "DAGs", "LMCs_data")
    file_name = f"δ vs. sqrt(height).png"
    file_path = join(parent_path, file_name)
    plt.savefig(file_path)
    plt.show()
    return

def δbyl0_vs_heightbyl0(df):

    
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
  
    
    
    x = df['height_over_l0']
    y = df['delta_over_l0']
    
    fig2, ax = plt.subplots(figsize=(18,12), dpi=400)
    ax.scatter(x, y, s=20)
    for xi, yi, Ni, rhoi, Li in zip(
        x, y, df['N'], df['rho'], df['L_over_l0']
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

def δbyl0_vs_sqrtheightbyl0(df):
    
    Height = df['height_over_l0']
    Delta = df['delta_over_l0']
    
    X = np.sqrt(Height)
    
    fig3, ax = plt.subplots(figsize=(18,12), dpi=400)
    ax.scatter(X, Delta, s=20)
    
    for xi, yi, Ni, H in zip(
        X, Delta, df['N'], df['height']
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
    
if __name__ == "__main__":
    
    # Read the files and create data frame
    all_deltas = all_delta_data()
    
    df = pd.DataFrame(all_deltas, columns=[
        'height','N','rho','l0','delta','delta_over_l0','L_over_l0','height_over_l0'
        ])
    # Plot 2 of δ vs. sqrt(height)
    δ_vs_sqrtheight(df)
    
    # Plot 3 of δ/l₀ vs. height/l₀ on a scatter plot
    δbyl0_vs_heightbyl0(df)
    
    # Plot 4 of δ/l_0 vs. \sqrt(height/l_0)
    δbyl0_vs_sqrtheightbyl0(df)
  
    
    
    
    
    # Step 3: Process multiple N values and generate the plot
    
    # Choose data set based on height and N values
    # height = 1000
    # most_frequent_sprinkling = True
    # min_freq = 4
    # Call the delt_vs_N function
    #delta_vs_N(height, most_frequent_sprinkling, min_freq)

    # Step 5: Finding relation between delta and N:
    
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













