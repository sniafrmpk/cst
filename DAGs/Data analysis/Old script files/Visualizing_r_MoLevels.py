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

def r_analyzer(N):
    parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "DAGs", "LMCs_data")
    
    # # Create a unique folder for this process
    folder_name = "Second batch"
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
            continue # very important, makes sure (0,0,0) is not added
        if t==1 and r==0:
             if sprinkling:  # Add the current chunk to the list
                 sprinklings.append(sprinkling)
        else:
            sprinkling.append((level, t, r))
    
    # Step 1: Group sprinklings by number of levels
    sprinkling_groups = defaultdict(list)

    for sprinkling in sprinklings:
        num_of_levels = len(set(level for level, t, r in sprinkling)) #length of the set containing all level values for that sprinkling. Same as max of level
        sprinkling_groups[num_of_levels].append(sprinkling)
    
    
    ######## Two kinds of data: 
        # 1. MoMoM: Compute mean of r and std error for each group sprinklings with unique num of chains
        #           by using a Mean of means of means approach. This prevents levels that have many nodes to overpower the mean for the sprinkling
        # 2. Level means for each group of sprinkling
    results_MoMoM = []
    chunk_means = []
    
    # Separate out the sprinkling groups for a given N
    for num_levels, chunk_list in sprinkling_groups.items(): # num_levels, chunk_list = path length, sprinklings with that length
        chunk_means = []
        chunk_level_means_list = []
        # Separate out the sprinklings for each group
            # Find mean for each level of each sprinkling to make mean for sprinkling. 
        for chunk in chunk_list:
            #num_levels = max(level for level, _, _ in chunk)  # Find max level
            all_levels = [[] for _ in range(num_levels)]            
            
            for level, t, r in chunk:
                if len(all_levels) < level: all_levels.append([])
                all_levels[level-1].append(r) # creates a lists of nodes at each level 
            
            level_means = [np.mean(rs) for rs in all_levels] # stores mean for each level of the sprinkling
            
            ######## From here on the two data sets will differ: ##########3
            
            ## MoMoM:
            # Compute mean and std error for each sprinkling ignorant of the level
            mean_r_sprinkling = np.mean(level_means)
            # se_r = np.std(all_r_values) # Don't know what to do with the std error for each sprinkling
            chunk_means.append(mean_r_sprinkling)
            
            ## level means
            chunk_level_means_list.append(np.array(level_means))
            
            
        ## MoMoM
        # Calculate the mean of the means for that group of sprinklings
        mean_of_means = np.mean(chunk_means)
        std_error = np.std(chunk_means)
    
        ## Level means
        # for i, lvl_means in enumerate(chunk_level_means_list):
        #     print(f"Sprinkling {i}: {len(lvl_means)} levels")
        group_level_means_list = np.array(chunk_level_means_list)
        group_level_means = np.mean(group_level_means_list, axis=0)
        group_level_se = np.std(group_level_means_list, axis=0)
       
        # Store the relevant results for each group of sprinklings as a tuple in results array
        results_MoMoM.append((N, num_levels, round(mean_of_means, 3), round(std_error,3), group_level_means, group_level_se, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
    
    
    return results_MoMoM


# Step 3: Process multiple N values and generate the plot
N_values = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 200000, 300000]  # Example list of N values

all_results = []

for N in N_values:
    all_results.extend(r_analyzer(N))

# Convert results to a DataFrame for easy plotting
df_results = pd.DataFrame(all_results, columns=["N", "num_levels", "mean_r", "SE_r", "group_level_means", "group_level_se", "num_chunks"])

# # Step 4: Plot mean r vs N with error bars
# plt.figure(figsize=(18, 12), dpi=300)

# for N in N_values:
#     subset = df_results[df_results["N"] == N] # It is a filtered DataFrame that contains only the rows from df_results where the column "N" equals the current N value in the loop.
    
# ##### Plot 1: MoMoM
        
#     most_frequent_sprinkling = True
#     # To plot groups with the most sprinklings
#         # Find the maximum num_chunks for the current N
#     if most_frequent_sprinkling:
#         all_sprinklings_gt_10 = False
#         off_set = 0.001
#         max_chunks = subset["num_chunks"].max()
#         # Filter the subset to only include rows where num_chunks is the maximum for this N
#         subset_filtered = subset[subset["num_chunks"] == max_chunks]
    
#     # Filter the subset to only include rows where num_chunks >= 10
#     else: 
#         all_sprinklings_gt_10 = True 
#         off_set = 0.002
#         min_freq = 10
#         subset_filtered = subset[subset["num_chunks"] >= min_freq]
    
#     # Plot
#     plt.errorbar(
#         [N] * len(subset_filtered), subset_filtered["mean_r"], xerr=subset_filtered["SE_r"], fmt='o', capsize=5
#     )
# # Adding labels with the tuple (num_levels, num_chunks)
#     for i, row in subset_filtered.iterrows():
#         label = f"({int(row['num_levels'])}, {int(row['num_chunks'])})"
#         plt.text(
#             N+0.1, row["mean_r"]+off_set, label, 
#             fontsize=9, ha='center', va='center', 
#             color='black', alpha=0.7
#         )
# plt.figtext(0.2, 0.90, "Tuples represent (chain length - 1, # of sprinklings with that length)", fontsize=10, ha='left', va='top', color='black')
       
# plt.xlabel("N")
# plt.ylabel("Mean r")
# if most_frequent_sprinkling:
#     title = "Mean r vs N - most frequent chain size"
#     file_name = "MoMoM_Most_freq_sprinkling_100.png"
# else:
#     title = f"Mean r vs N - freq(any chain size) >= {min_freq}"
#     file_name = f"MoMoM_Sprinklings_gt_{min_freq}_100.png"
# plt.title(title, y=1.03)
# plt.savefig(file_name)
# plt.show()

# # ============================
# # NEW PLOT: group_level_means vs Level Index for each N
# # ============================
#plt.figure(figsize=(18, 12), dpi=300)

for N in N_values:
    subset = df_results[df_results["N"] == N]  # Filter rows for current N
    ## Filtering based on freq
    # min_freq = 10
    # subset_filtered = subset[subset["num_chunks"] >= min_freq]
    
    ## Filtering the most common
    max_chunks = subset["num_chunks"].max()
    subset_filtered = subset[subset["num_chunks"] == max_chunks]

    y_min = 0.01
    y_max = 0.2
    for _, row in subset_filtered.iterrows():
        group_level_means = np.array(row["group_level_means"])  # Convert to NumPy array
        group_level_se = np.array(row["group_level_se"])  # Convert to NumPy array
        levels = np.arange(1, len(group_level_means) + 1)  # Levels (indices)
        
        # Create the plot for r vs level with error bars
        plt.figure(figsize=(16, 10))
        plt.errorbar(levels, group_level_means, yerr=group_level_se, fmt='o', capsize=5, label=f'N={N}')
        plt.ylim(y_min, y_max)
        plt.xlabel('Level')
        plt.ylabel('Mean r')
        plt.title(f'Mean r vs Level for N = {N}')
        plt.legend()
        plt.grid(True)
        # # Create a unique folder for this process
        parent_path = join(expanduser("~"), "Desktop", "GitRepos", "cst_longest_maximal_chains", "DAGs", "Plots for 100 sprinklings")
        file_name = f"N{N}_#levels_{levels.max()}.png"
        file_path = join(parent_path, file_name)
        os.makedirs(parent_path, exist_ok=True)
        plt.savefig(file_path)
        
        plt.show()
#         plt.errorbar(
#             levels, group_level_means, yerr=group_level_se, label=f"N={N}, Levels={row['num_levels']}",
#             fmt='-o', capsize=5, alpha=0.7
#         )

# plt.xlabel("Level Index")
# plt.ylabel("Mean r at Level")
# plt.title("Mean r at Each Level for Different N Values")
# plt.legend(loc='best', fontsize=9, ncol=2, frameon=True)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.savefig("group_level_means_plot.png")
# plt.show()

# sg, all_results = r_analyzer(10000)
# # # means_of_means = []
# # # list_of_std_errors = []

# # # for N in [5000, 10000, 20000, 30000, 40000, 50000, 65000, 80000]:
# # #     mean_of_means, std_error = r_analyzer(N)
# # #     means_of_means.append(mean_of_means)
# # #     list_of_std_errors.append(std_error)

# # # Ns = [5000, 10000, 20000, 30000, 40000, 50000, 65000, 80000]

# # # plt.plot(Ns, means_of_means)
# # # plt.show()












