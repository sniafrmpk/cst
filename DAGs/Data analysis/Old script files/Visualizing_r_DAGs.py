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

def r_analyzer(N):
    # df = pd.read_csv(f"spherical_coordinates_N{N}.csv", skiprows=1)
    # r = df.iloc[:, 1] # all rows of column 1
    
    #   r = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Convert to numeric
    # Split the file into sprinklings by finding rows where r is 0
    sprinklings = []
    sprinkling = []
    
    df = pd.read_csv(f"spherical_coordinates_N{int(N/1000)}k.csv", usecols=[1, 2, 3])
    # Iterate over rows
    for row in df.itertuples(index=False):  # `index=False` avoids including index in tuple
        level, t, r = row  # Unpack the tuple
        if t == 0: 
            sprinkling = [] 
        if t==1:
             if sprinkling:  # Add the current chunk to the list
                 sprinklings.append(sprinkling)
        else:
            sprinkling.append((level, t, r))
    
    # Step 1: Group sprinklings by number of levels
    sprinkling_groups = defaultdict(list)

    for sprinkling in sprinklings:
        num_of_levels = len(set(level for level, t, r in sprinkling)) #length of the set containing all level values for that sprinkling. Same as max of level
        sprinkling_groups[num_of_levels].append(sprinkling)
    
    # # Get the number of items stored for each key
    # for key, values in sprinkling_groups.items():
    #     print(f"num_of_levels: {key}, Number of sprinklings: {len(values)}")
    
    # Step 2: Compute mean of r and std error for each group sprinklings with unique num of chains
    results = []
    chunk_means = []
    #group_means = {} # For more fine grained statistics for later

    for num_levels, chunk_list in sprinkling_groups.items():#path length, sprinklings with that length
        #level_r_values = defaultdict(list)
        all_r_values = []
        chunk_means = []
    # Collect all r-values for each sprinkling regardless of the level
    # can also do for level using level_r_values
        for chunk in chunk_list:
            all_r_values.extend([r for level, t, r in chunk])
            # for level, t, r in chunk:
            #     level_r_values[level].append(r)

            # Compute mean and std error for each sprinkling ignorant of the level
            mean_r = np.mean(all_r_values)
            
            # se_r = np.std(all_r_values) # Don't know what to do with the std error for each sprinkling
            chunk_means.append(mean_r)
            
            # reset all_r_values
            all_r_values = []
            
        # Compute mean and standard error of r for each level
        # level_means = {level: np.mean(r_list) for level, r_list in level_r_values.items()}
        # level_se = {level: np.std(r_list, ddof=1) for level, r_list in level_r_values.items()}
        # #group_means[num_levels] = level_means  # Store level-wise means for the group
        
        # Calculate the mean of the means for that group of sprinklings
        mean_of_means = np.mean(chunk_means)
        std_error = np.std(chunk_means)
        # Store the relevant results for each group of sprinklings as a tuple in results array
        results.append((N, num_levels, round(mean_of_means, 3), round(std_error,3), len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length

        # for level, mean_r in level_means.items():
        #     results.append((N, num_levels, level, round(mean_r, 3), round(level_se[level], 3), len(chunk_list)))
        
    # Print the results
    # for num_levels, mean_value in final_means.items():
    #     print(f"Mean of means for sprinklings with {num_levels} levels: {mean_value}")
    # # Calculate the mean of each chunk
    # means = [np.mean(chunk) for chunk in chunks]
   
    # # Calculate the mean of the means
    # mean_of_means = np.mean(means)

    # # Calculate the standard error of the means
    # std_error = np.std(means)

    # print(f"Mean of means for N = {N}:", mean_of_means)
    # print(f"Standard error of the means for N = {N}:", std_error)

    # return mean_of_means, std_error
    return results


# Step 3: Process multiple N values and generate the plot
N_values = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000]  # Example list of N values

all_results = []

for N in N_values:
    all_results.extend(r_analyzer(N))

# Convert results to a DataFrame for easy plotting
df_results = pd.DataFrame(all_results, columns=["N", "num_levels", "mean_r", "SE_r", "num_chunks"])

# Step 4: Plot mean r vs N with error bars
plt.figure(figsize=(8, 6), dpi=300)

for N in N_values:
    subset = df_results[df_results["N"] == N]
    
    # To plot the ones with the most sprinklings
    # # Find the maximum num_chunks for the current N
    max_chunks = subset["num_chunks"].max()
    
    # Filter the subset to only include rows where num_chunks is the maximum for this N
    subset_filtered = subset[subset["num_chunks"] == max_chunks]
    
    # Filter the subset to only include rows where num_chunks >= 10
    #subset_filtered = subset[subset["num_chunks"] >= 10]
    
    # Plot
    plt.errorbar(
        [N] * len(subset_filtered), subset_filtered["mean_r"], xerr=subset_filtered["SE_r"], fmt='o', capsize=5
    )
# Adding labels with the tuple (num_levels, num_chunks)
    for i, row in subset_filtered.iterrows():
        label = f"({int(row['num_levels'])}, {int(row['num_chunks'])})"
        plt.text(
            N+0.1, row["mean_r"]+0.001, label, 
            fontsize=9, ha='center', va='center', 
            color='black', alpha=0.7
        )
plt.figtext(0.2, 0.90, "Tuples represent (chain length - 1, # of sprinklings with that length)", fontsize=10, ha='left', va='top', color='black')
       
plt.xlabel("N")
plt.ylabel("Mean r")
plt.title("Mean r vs N with chain length differentiated", y=1.03)
#plt.grid(True)
plt.savefig("high_res_plot.png")
plt.show()



# r_analyzer(5000)
# # means_of_means = []
# # list_of_std_errors = []

# # for N in [5000, 10000, 20000, 30000, 40000, 50000, 65000, 80000]:
# #     mean_of_means, std_error = r_analyzer(N)
# #     means_of_means.append(mean_of_means)
# #     list_of_std_errors.append(std_error)

# # Ns = [5000, 10000, 20000, 30000, 40000, 50000, 65000, 80000]

# # plt.plot(Ns, means_of_means)
# # plt.show()












