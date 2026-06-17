#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:28:11 2024
"""
import pandas as pd
import numpy as np
import ast
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict 
import os
from pathlib import Path
import plotly.express as px

def read_path(D, height):
    folder_path = Path(__file__).parent.parent / "LMCs_data" / "Intervals" / f"D {D} - Height {height}"
    os.makedirs(folder_path, exist_ok=True)
    return str(folder_path)

def write_path():
    return str(Path(__file__).parent)

# ---------------------------------------------------------------------
# Data collection pipelines (quick reference)
#
# Pipeline 1 (summary delta table for delta/H vs N):
#   delta_analyzer_middle_layers -> all_delta_data -> df_delta_data_all
#   Uses (level, t, r), groups by k_max, computes per-group mean_of_means,
#   then delta = mean over groups.
#
# Pipeline 2 (bootstrap/error-bar input):
#   all_sprinklings_tuples_for_D_H_N -> all_deltas -> bootstrap_bounds / deltabyH_vs_N
#   Uses (level, t, r), computes one delta per sprinkling (mean r), returns
#   list-of-arrays indexed by N.
#
# Pipeline 3 (shape/time visualization):
#   sprinklings_D_H_N -> data_rs_ts_all -> δ_vs_t / num_paths tooling
#   Uses full row (element, level, t, r, theta, phi, preds), then flattens to
#   rs/ts/thetas/phis per k_max group for plotting.
#
# Why 1 and 3 are not identical outputs:
#   Pipeline 3 flattens point clouds for visualization and path-shape analysis.
#   It does not preserve the per-sprinkling/per-level structure used by
#   Pipeline 1's mean_of_means -> delta aggregation.
#
# Unification direction (if you refactor later):
#   Build one canonical loader that returns sprinklings with optional fields
#   (minimal: level,t,r; full: +element,theta,phi,preds). Then keep separate
#   "reducers" for summary-delta, bootstrap arrays, and shape plots.
# ---------------------------------------------------------------------

######### how many sprinklings ##########
# how is it different to sprinklings_D_H_N() defined below?
def all_sprinklings_tuples_for_D_H_N(D, height, N, folder_path):
        """
        for now collects level, t and r as tuples for all sprinklings of a given D, H, N in a 
        list of list called sprinklings

        """
        file_name = f"spherical_coordinates_D{D}_H{height}_N{int(N/1000)}k.csv"
        file_path = join(folder_path, file_name)

        # Read only the level, t and r from each row.
        df = pd.read_csv(file_path, usecols=[1, 2, 3])

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
        num_of_sprinklings = len(sprinklings)
        print(f"Number of sprinklings for D={D}, N={N}, height={height}: {num_of_sprinklings}")
        return sprinklings

def num_of_sprinklings(Ds, heights, Ns, folder_path):
    """
        helper function to print (maybe later store,) how many sprinklings for a given data set
    """
    for D in Ds:
        for height in heights:
            ## This is the loop in which we gather all the results 
            # sequentially for each N by calling delta_analyzer_all_levels(N)
            for N in Ns:
                all_sprinklings_for_D_H_N(D, height, N, folder_path) # returns sprinklings as a list but since its not assigned to any variable, it is cleared, only the print function is used.

######### Collecting data ######
# Pipeline 1: produces summary statistics used in df_delta_data_all.
def delta_analyzer_middle_layers(folder_path, D, height, N, use_middle_third, file_path_override=None):
    """

    Parameters
    ----------
    N : int
        Number of points in the sprinkling.
    height : int
        The hieght of the sprinkled region.
    use_middle_third : bool
        If True, delta for a sprinkling is calculated using the middle third in time:
        points with t in [height/3, 2*height/3].
    
    Returns
    -------
    results : List
            Stores the following for each group of sprinklings with the same N, H and k_max = path length
        (height, N, path_length, round(mean_of_sups, 4), round(std_sups, 4), round(mean_of_means, 4), round(std_error,4), group_level_means, group_level_se, group_level_means_t, group_level_se_t, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
        Here, mean_of_means is a per-group quantity: it averages sprinkling-level deltas
        only within one path-length group (fixed k_max).

    delta_data : Array with one row
    Stores the following for all the sprinklings with the same H and N.    
    [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    Where delta is an across-group quantity: after computing mean_of_means for each
    path-length group, delta is the mean of those group means for the same N and H.
    Therefore, each path-length group has equal weight in delta, regardless of how
    many sprinklings are in that group.
    
    """
    # File name (can be overridden for non-standard filenames)
    if file_path_override is None:
        file_name = f"spherical_coordinates_D{D}_H{height}_N{int(N/1000)}k.csv"
        file_path = join(folder_path, file_name)
    else:
        file_path = file_path_override

    # Read only the level, t and r from each row.
    df = pd.read_csv(file_path, usecols=[1, 2, 3])

    # If use_middle_third is truthy, use the middle third in time.
    if use_middle_third:
        middle = True
    else:
        middle = False
    
    
    # Step 1: Split the file into sprinklings by finding rows where r is 0
    sprinkling_groups = defaultdict(list)

    current_inside = []
    current_k_max = None

    for level, t, r in df.itertuples(index=False, name=None):
        level = int(level)
        t = float(t)
        r = float(r)

        # start of a new sprinkling
        if level == 0:
            current_inside = [] # don't store s
            current_k_max = None
            continue

        # target row → defines k_max, target is not stored
        if t == height:
            current_k_max = level
            # finalize sprinkling
            if current_k_max is not None:
                sprinkling_groups[current_k_max].append(current_inside)
            current_inside = []
            current_k_max = None
            continue

        # inside point
        current_inside.append((level, t, r))

    # Step 2: Compute mean of r and std error for each group sprinklings with unique num of chains
    # by using a Mean of means of means approach. This prevents levels that have many nodes to overpower the mean for the sprinkling
    
    # A chunk is a sprinkling, so called because it is a 'chunk' of rows in the csv file.
    
    results = []
    chunk_means = []
    
    # This variable is supposed to disregard the fact that sprinkling groups have varrying
    # path length. It just cares about height of the sprinkled region and the number of points.
    # For all sprinklings with the same height and N, it stores the mean of delta for the middle
    # three layers. Later on, we take a mean of these to get a represenative delta for all the 
    # sprinklings with a given height and N.
    deltas_middle_layers = []
    
    for k_max, chunk_list in sprinkling_groups.items(): # path length, sprinklings with that length
        group_means = []
        supremums = [] # contains arrays of the [sup_r, sup_level, num_of_levels] for each sprinkling

        # array to store the average r for each level for all sprinklings of the same path length
        chunk_levels_means_list = []
        chunk_levels_means_list_t = []
        
    # Find mean for each level of each sprinkling to make mean for sprinkling. 
        num_levels_inside = k_max - 1 # excludes s and t
        for chunk in chunk_list:
            all_levels_inside = [[] for _ in range(num_levels_inside)] # stores the nodes at each level. Basically the all_levels array from the og code
            all_levels_inside_t = [[] for _ in range(num_levels_inside)]
            sup_r, sup_level = 0, 0
            for level, t, r in chunk:
                #print(f"level = {level}, len(all_levels) = {len(all_levels)}")
                #if len(all_levels) < level: all_levels.append([])
                level = int(level)
                r = float(r)
                if 1 <= level <= k_max - 1: # with this condition, even if I store s and t in the chunk, they are ignored here
                    if r > sup_r:
                        sup_r = r
                        sup_level = level

                    idx = level - 1  # level 1 -> inside_level 0
                    all_levels_inside[idx].append(r)
                    all_levels_inside_t[idx].append(t)
            
            # 0A. 
            # Append supremum data for this sprinkling
            supremums.append((sup_r, sup_level))
           
            # 2A.
   ########## Step 1 of determining delta: list of mean delta for each level of the sprinkling
            sprinkling_means = [np.mean(rs) if rs else np.nan for rs in all_levels_inside] # stores mean for each level of the sprinkling
            # array of means of all levels of all sprinklings of the same chain length
            chunk_levels_means_list.append(np.array(sprinkling_means))
            
            # 3A.
            sprinkling_means_t = [np.mean(ts) if ts else np.nan for ts in all_levels_inside_t]
            chunk_levels_means_list_t.append(np.array(sprinkling_means_t))

            #### 1A.
   ########## Step 2 of determining delta:
            if not middle:
                # Compute mean and std error for each sprinkling ignorant of the level
                mean_r = np.nanmean(sprinkling_means)
                # se_r = np.std(all_r_values) # Don't know what to do with the std error for each sprinkling
            else:
                # Use the middle third in time: t in [height/3, 2*height/3]
                t_low = height / 3.0
                t_high = 2.0 * height / 3.0
                slab_means = []
                for rs, ts in zip(all_levels_inside, all_levels_inside_t):
                    if not rs:
                        slab_means.append(np.nan)
                        continue
                    # filter points in this level by time slab
                    rs_in_slab = [r for r, t in zip(rs, ts) if t_low <= t <= t_high]
                    slab_means.append(np.mean(rs_in_slab) if rs_in_slab else np.nan)
                mean_r = np.nanmean(slab_means)
    
   ########## Step 3 of determining delta:
            # Store delta for this sprinkling in the array group_means representing the array of means for all the sprinklings with the same longest chain length
            group_means.append(mean_r)
            
            
            # reset arrays for this sprinkling (not strictly necessary due to reinit above)
            all_levels_inside = []
            all_levels_inside_t = []
            
        # 0B. Calculate the mean of the supremums for that group of sprinklings
        mean_of_sups = np.mean(np.array(supremums), axis=0)
        std_sups = np.std(np.array(supremums), axis=0)
        
        # 1B. Calculate the mean of the means for that group of sprinklings

######### Step 4 of determining delta:
        # mean_of_means is per k_max group (one path-length group only).
        mean_of_means = np.nanmean(group_means)
        # std_error here is over path-length groups; for per-sprinkling error bars use all_deltas() below.
        std_error = np.nanstd(group_means)
        
        # 2B. Converts to a 2D array of shape (sprinkling number, level number)
        group_level_means_list = np.array(chunk_levels_means_list)
        group_level_means = np.mean(group_level_means_list, axis=0)
        group_level_se = np.std(group_level_means_list, axis=0)
        
        # 3B.
        group_level_means_list_t = np.array(chunk_levels_means_list_t)
        group_level_means_t = np.mean(group_level_means_list_t, axis=0)
        group_level_se_t = np.std(group_level_means_list_t, axis=0)
        
        # Store the relevant results for each group of sprinklings as a tuple in results array
        path_length = k_max
        results.append((height, N, path_length, np.round(mean_of_sups, 4), np.round(std_sups, 4), round(mean_of_means, 4), round(std_error,4), group_level_means, group_level_se, group_level_means_t, group_level_se_t, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
        
######### Step 5 of determining delta:
        # An array that stores the deltas for each group of sprinklings 
        deltas_middle_layers.append(mean_of_means) 
        #print(results)
   
######### Step 6 of determining delta:
    ## 3. average delta for all the sprinklings
    # delta is across k_max groups: equal weight per path-length group.
    delta = np.mean(deltas_middle_layers)  
    
    # The array / list I want to be returned
    ##### For intervals only!
    # l_0 = (height / N) ** 0.25 # L = 1
    # L = 1
    # rho = N / (height * L**3)
    if D == 4:
        V = np.pi/24*height**4  # True for D = 4
    if D == 3:
        V = np.pi/12*height**3  # True for D = 3
    elif D == 2:
        V = 0.5*height**2  # True for D = 2
    rho = N/V
    l_0 = rho**(-1/D)  # General formula for any D

    #delta_data = [height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    delta_data = [D, height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / height, 3), round(delta / l_0, 3), round(height / l_0, 3)]
    return results, delta_data

# Pipeline 2: one delta value per sprinkling, grouped by N (for bootstrap/error bars).
# delta is defined here as the mean r over all inside points in that sprinkling.
def all_deltas(D, H, Ns, def_delta_all_elements, folder_path, use_middle_third=False):
    """
    creates a python list of numpy arrays, each of which for one sprinkling ensemble.
    This is the helper used by all_delta_data(..., delta_weighting="equal_sprinkling").
    If use_middle_third=True, each sprinkling delta is computed from points with
    t in [H/3, 2H/3] only; otherwise all inside points are used.
    """
    # Collect all sprinklings
    all_deltas_list = []
    for N in Ns:
        sprinklings_list = all_sprinklings_tuples_for_D_H_N(D, H, N, folder_path)
        # for the definition of delta that uses r-values of ALL elements
        if def_delta_all_elements:
            # array of deltas for all sprinklings with same D, H, N 
            # using the all_elements definition of delta which is blind to the path length and
            # how many paths share an element.
            if use_middle_third:
                t_low = H / 3.0
                t_high = 2.0 * H / 3.0
                sps_mean_rs = []
                for sprinkling in sprinklings_list:
                    rs_middle = [r for _, t, r in sprinkling if t_low <= t <= t_high]
                    # Keep NaN if a sprinkling has no points in the middle-third slab.
                    sps_mean_rs.append(np.mean(rs_middle) if rs_middle else np.nan)
            else:
                sps_mean_rs = [np.mean([r for _,_,r in sprinkling]) for sprinkling in sprinklings_list]
            
            # append that array to a list that contains similar arrays for differing N values.
            all_deltas_list.append(sps_mean_rs)
    return all_deltas_list

# Collect data using delta_analyzer_middle_layers over all Ds, heights and N values and returns a dataframe
def all_delta_data(use_middle_third, Ds, heights, N_values, delta_weighting="equal_path_length"):
    """
    Runs loops over values of height and N to collect all the lists
    delta_data = [D, height, N, int(rho), round(l_0, 3), round(delta, 3), round(delta / l_0, 3), round(L/l_0, 3), round(height / l_0, 3)]
    with L = 1 and all values rounded to 3 dp, in a master list all_delta_data.

    Parameters
    ----------
    use_middle_third : bool
        Used only when delta_weighting="equal_path_length" (Pipeline 1).
    delta_weighting : str
        "equal_path_length" -> current Pipeline 1 behavior (mean over k_max groups).
        "equal_sprinkling" -> Pipeline 2 behavior (mean of per-sprinkling deltas).
        Quick usage:
        - all_delta_data(False, [4], [1, 10], Ns, "equal_path_length")
        - all_delta_data(False, [4], [1, 10], Ns, "equal_sprinkling")
    
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
    for D in Ds:
        for height in heights:
            # N values
            # if height == 1:
            #     N_values = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 200000, 300000, 400000]
            # else:
            # N_values = [10000, 20000, 40000, 80000, 160000, 320000]  # List of N values
            # N_values = np.array(N_values)
            
            folder_path = read_path(D, height)
            if delta_weighting == "equal_path_length":
                # Sequentially for each N by calling delta_analyzer_middle_layers (Pipeline 1)
                # Weighting rule in this branch:
                #   delta = mean over k_max groups of (mean delta within that group)
                for N in N_values:
                    # Store only delta_data tuple using [1]
                    results_N_height, delta_data_N_height = delta_analyzer_middle_layers(
                        folder_path, D, height, N, use_middle_third
                    )

                    # Iteratively store all the delta_data tuples
                    delta_data_all.append(delta_data_N_height)
                    results_all.append(results_N_height)

            elif delta_weighting == "equal_sprinkling":
                # Pipeline 2: one delta per sprinkling, then average with equal sprinkling weight
                # IMPORTANT: this branch does not regroup by k_max/path length.
                # Each sprinkling contributes one delta and gets equal weight.
                all_deltas_list = all_deltas(
                    D, height, N_values, True, folder_path, use_middle_third=use_middle_third
                )
                for N, deltas_for_N in zip(N_values, all_deltas_list):
                    delta = float(np.nanmean(np.asarray(deltas_for_N, dtype=float)))

                    if D == 4:
                        V = np.pi / 24 * height**4
                    elif D == 3:
                        V = np.pi / 12 * height**3
                    elif D == 2:
                        V = 0.5 * height**2
                    else:
                        raise ValueError(f"Unsupported D={D} for interval volume.")

                    rho = N / V
                    l_0 = rho**(-1 / D)
                    delta_data_all.append([
                        D, height, N, int(rho), round(l_0, 3), round(delta, 3),
                        round(delta / height, 3), round(delta / l_0, 3), round(height / l_0, 3)
                    ])
            else:
                raise ValueError(
                    "delta_weighting must be 'equal_path_length' or 'equal_sprinkling'."
                )
            
    # create dfs
    df_delta_data = pd.DataFrame(delta_data_all, columns=[
        'D','height','N','rho','l0','delta', 'delta_over_H', 'delta_over_l0','height_over_l0'
        ])
    # Stored so plotting functions can auto-label titles/files with the delta definition used.
    if delta_weighting == "equal_sprinkling":
        if use_middle_third:
            df_delta_data["delta_def"] = "middle third in time, equal sprinkling weight"
        else:
            df_delta_data["delta_def"] = "all elements in time, equal sprinkling weight"
    else:
        df_delta_data["delta_def"] = "middle third in time, equal path length weight" if use_middle_third else "all levels in time, equal path length weight"
    
    # df_results = pd.DataFrame(results_all, columns=["Height", "N", "num_levels", "mean_r", "SE_r", "group_level_means", "group_level_se", "group_level_means_t", "group_level_se_t", "num_sprinklings"])


    #return df_delta_data, df_results
    return df_delta_data


######### Delta vs. N ######
# Original function
def δbyH_vs_N(Ds, df_delta_data_all, delta_definition=None, annotate_points=True, annotation_fontsize=7):
    """
    df_delta_data = pd.DataFrame(all_deltas, columns=[
        'D','height','N','rho','l0','delta', 'delta_over_H', 'delta_over_l0', 'height_over_l0'
        ])

    delta_definition: Optional label or flag for how delta was computed.
        - If bool: True -> "middle third in time", False -> "all levels in time"
        - If str: used directly
        - If None: attempts to infer from df_delta_data_all["delta_def"] if present
    annotate_points: If True, write a label next to each point as (delta/H, H/l0).
    annotation_fontsize: Font size used for point labels.
    """
    # Determine a human-readable delta definition label
    if delta_definition is None:
        if "delta_def" in df_delta_data_all.columns:
            defs = [str(d) for d in df_delta_data_all["delta_def"].dropna().unique()]
            delta_def_label = defs[0] if len(defs) == 1 else f"mixed ({', '.join(defs)})"
        else:
            delta_def_label = "unspecified definition"
    elif isinstance(delta_definition, bool):
        delta_def_label = "middle third in time" if delta_definition else "all levels in time"
    else:
        delta_def_label = str(delta_definition)

    # Safe suffix for filenames (ASCII-ish)
    delta_def_suffix = "".join([c if c.isalnum() else "_" for c in delta_def_label]).strip("_") or "unspecified"

    for D in Ds:
        df_delta_data = df_delta_data_all[df_delta_data_all['D'] == D].sort_values(['height', 'N'])
        print(df_delta_data)
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
        groups_by_height = df_delta_data.groupby('height')
        fit_results = {}

        for H, group in groups_by_height:
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
            ax2.set_title(f'Power-law: δ = A N^(-β) for D={D}, H={H}\nΔ definition: {delta_def_label}')
            #plt.title('Power-law: δ = A N^(-β), Exp-decay: δ = A exp(-alpha*N)')
            ax2.legend()
            ax2.grid(True)
            fig2.tight_layout()
            parent_path = write_path()
            folder_name = f"DeltabyH vs. N"
            folder_path = join(parent_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            file_name = f'Power-law fit - D {D} H {H} - {delta_def_suffix}.png'
            file_path = join(folder_path, file_name)
            fig2.savefig(file_path)
            plt.close(fig2)
        # Annotate points with (delta/H, H/l0). Alternate offsets to reduce overlap.
        if annotate_points:
            df_annot = df_delta_data.reset_index(drop=True)
            for idx, row in df_annot.iterrows():
                xi = row['N']
                yi = row['delta'] / row['height']
                H = row['height']
                l0 = row['l0']
                dx = 6 if idx % 2 == 0 else -6
                dy = 6 if idx % 3 else -8
                ax.annotate(
                    f"({yi:.3f}, {H/l0:.2f})",
                    xy=(xi, yi),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha='left' if dx > 0 else 'right',
                    va='bottom' if dy >= 0 else 'top',
                    fontsize=annotation_fontsize,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.55)
                )
        
        ax.set_xlabel('N', fontsize=20)
        ax.set_ylabel('δ/H', fontsize=20)
        if annotate_points:
            title_main = 'δ/H vs. N with point labels: (δ/H, H/l₀)'
        else:
            title_main = 'δ/H vs. N'
        ax.set_title(f'{title_main}\nΔ definition: {delta_def_label}', fontsize=20)
        ax.legend(fontsize=12, loc='best')

        fig1.tight_layout()
        parent_path = write_path()
        folder_name = "DeltabyH vs. N"
        folder_path = join(parent_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"δbyH vs. N - D {D} - {delta_def_suffix}.png"
        file_path = join(folder_path, file_name)
        fig1.savefig(file_path)
        plt.show()
    
    return

# New function with error bars. Uses all_deltas() defined above.
def deltabyH_vs_N(all_deltas_list, D, H, Ns):
    """
    Docstring for deltabyH_vs_N
    
    :param all_deltas_list: list of arrays of all deltas for all sprinklings from all_deltas()
    :param Ns: Same Ns that go into all_deltas()

    :return plot of deltabyH vs. N
    """
    # create one delta for each N:
    deltas_for_each_N = np.asarray([np.mean([delta for delta in all_deltas_N]) for all_deltas_N in all_deltas_list])
    # create a standard deviation value using deltas for all sprinklings for each N
    std_for_each_N = np.asarray([np.std([delta for delta in all_deltas_N]) for all_deltas_N in all_deltas_list])

    ### create figure
    # data to plot
    N_values = Ns
    delta_over_H = deltas_for_each_N/H

    # figure
    fig1, ax = plt.subplots(figsize=(20,10), dpi=400)
    # scatter for this height group
    ax.errorbar(
            N_values, delta_over_H, yerr=std_for_each_N, fmt='o', capsize=5, label=fr'Mean $\delta$ with std. dev. from sprinklings'
            )
    
    # log–log transform, log in python is ln
    X = np.log(N_values)
    Y = np.log(delta_over_H)

    # Linear fit: Y = slope*X + intercept, where slope = -beta and intercept = log A
    slope, intercept = np.polyfit(X, Y, 1)

    beta = -slope             # minus sign because slope is negative
    A = np.exp(intercept)     # prefactor

    # --- plot the fit curve with A and beta in legend ---
    N_fit = np.linspace(min(N_values), max(N_values), 200)
    y_fit = A * N_fit**(-beta)
    ax.plot(
        N_fit, y_fit, '--',
        label=fr'$A\,N^{{-\beta}}$: A={A:.3f}, β={beta:.3f}'
    )
            
    ax.set_xlabel('N', fontsize=20)
    ax.set_ylabel('δ/H', fontsize=20)
    ax.set_title(f'δ/H vs. N for D={D}, H={H}', fontsize=20)
    
    ax.legend(fontsize=12, loc='best')

    plt.tight_layout()
    parent_path = write_path()
    folder_name = "DeltabyH vs. N"
    folder_path = join(parent_path, folder_name)
    file_name = f"δbyH vs. N - D{D} - H{H}.png"
    file_path = join(folder_path, file_name)
    plt.savefig(file_path)

    return

######### error in \beta using non-parameterized bootstrap ###########
def bootstrap_bounds(all_deltas, Ns, B):
    """
    Docstring for bootstrap_bounds
    
    :param all_deltas: list of arrays containing delta/Hs for sprinklings based on N
    Ns
    B ~ 2000 the number of bootstrap samples
    return: bound_beta, bound_A: tuples of beta_boot_2.5%-97.5% ; similarly for A.
    """
    # Idea of bootstrap method:
    # for a each sprinkling set, all_deltas[k], characterized by N_k (N_0 = 654 e.g.), create new a set of the same size by sampling from all_deltas[k] with replacement.
    # Thus you will have a new all_deltas_B1. find beta_B1 and A_B1 for it. then create a new all_deltas_B2 from all_deltas like before, and find similarly calculate
    # beta_B2 and A_B2. Repeat this say 2000 times.
    # Then you will have two arrays for various values of beta_B and A_B. Find values within them such that only 2.5% of the values lie above and below them, these four values 
    # go in to bound_beta and bound_A
    
    boot_betas = []
    boot_As = []
    X = np.log(Ns)
    rng = np.random.default_rng()

    for b in range(B):
        # create new all_deltas = all_deltas_b
        all_deltas_b = []
        k = 0
        for N in Ns:
            # find num_of_sprinklings for that N
            T_k = len(all_deltas[k])
            
            # create a list of random indices uniformly to sample from all_deltas[k] to create all_deltas_b[k]
            idx = rng.integers(0, T_k, T_k) # upper bound exclusive, like we want since maximum index is T_k-1.

            # sample deltas at these indices from all_deltas[k]
            arr = np.asarray(all_deltas[k])
            all_deltas_b.append(arr[idx])
            k += 1
            
        # find beta_b and A_b for all_deltas_b using log-log fit
        mean_deltas_b_for_all_Ns = [np.mean([delta_b_k_i for delta_b_k_i in deltas_b_k]) for deltas_b_k in all_deltas_b]
        # log–log transform, log in python is ln
        Y = np.log(mean_deltas_b_for_all_Ns)

        # Linear fit: Y = slope*X + intercept, where slope = -beta and intercept = log A
        slope, intercept = np.polyfit(X, Y, 1)

        beta_b = -slope             # minus sign because slope is negative
        A_b = np.exp(intercept)     # prefactor
        
        # append them to boot_betas and boot_As
        boot_betas.append(beta_b)
        boot_As.append(A_b)
    
    # find lower and upper bounds on beta_b and A_b
    bound_beta = [np.percentile(boot_betas, 2.5), np.percentile(boot_betas, 97.5)]
    bound_A = [np.percentile(boot_As, 2.5), np.percentile(boot_As, 97.5)]
    return bound_beta, bound_A

######### Shape ########
# Pipeline 3: full-geometry loader used for rs/ts/theta/phi visualizations and path tools.
def sprinklings_D_H_N(D, height, N):
    """
    Docstring for sprinkling_groups
    
    :param N: Description
    return: sprinkling_groups: dict where key = num_of_levels, value = list of sprinklings with that num_of_levels
    """
    parent_path = read_path(D, height)
    # File name
    file_name = f"spherical_coordinates_D{D}_H{height}_N{int(N/1000)}k.csv"
    file_path = join(parent_path, file_name)
    
    # Read only the level, t and r from each row.
    def _parse_preds(preds):
        s = str(preds).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s:
            return []
        return [int(x) for x in s.split()]

    df = pd.read_csv(
        file_path,
        usecols=[0, 1, 2, 3, 4, 5, 6],
        converters={6: _parse_preds}
    ) # element, level, t, r, theta, phi, preds
    
    # Step 0: Split the file into sprinklings by finding rows where r is 0
    sprinkling = []
    num_of_levels = 0
    # Step 1: Group sprinklings by number of levels
    sprinkling_groups = defaultdict(list)
    
    # Iterate over rows
    for row in df.itertuples(index=False):  # `index=False` avoids including index in tuple
        element,level, t, r, theta, phi, preds = row  # Unpack the tuple
        if level == 0: 
            sprinkling = []
            sprinkling.append((element, level, t, r, theta, phi, preds)) # s and t are also added to the levels. s is at level 0 and t is at levels k_max (chain length)
            continue
        if t == height:
            num_of_levels = level
            sprinkling.append((element, level, t, r, theta, phi, preds))
            if sprinkling:  # Add the current chunk to the list
                sprinkling_groups[num_of_levels].append(sprinkling)
        else:
            sprinkling.append((element, level, t, r, theta, phi, preds))
            
    return sprinkling_groups

def data_rs_ts_all(D, height, N, sprinklings):
    """
    Parameters
    ----------
    D : int
        The dimension of the diamond.
    N : int
        Number of points in the sprinkling.
    height : int
        The height of the sprinkled region.

    Returns
    -------
    results : List
        Stores the rs and ts for all the points of all the sprinklings with the same N, H and k_max = path length 
        results.append((D, height, N, num_of_levels, rs, ts, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length

    """
    # Step 2: Working with groups of sprinklings with the same path length.
    # A chunk is a sprinkling, so called because it is a 'chunk' of rows in the csv file.
    # Note: this function intentionally flattens point data across sprinklings in each
    # group; good for visualization, but not a drop-in replacement for Pipeline 1
    # aggregation (which needs per-sprinkling/per-level structure).
    
    # the final array of results 
    results = []

    for num_of_levels, chunk_list in sprinklings.items(): # path length, sprinklings with that length
        ts = []
        rs = []
        thetas = []
        phis = []
        for chunk in chunk_list:
            for _, level, t, r, theta, phi, _ in chunk:
                # r and t are stored for all the sprinklings with the same path length
                rs.append(r) 
                ts.append(t)
                thetas.append(theta)
                phis.append(phi)

        # Store the relevant results for each group of sprinklings as a tuple in results array
        results.append((D, height, N, num_of_levels, rs, ts, thetas, phis, len(chunk_list))) #len(chunk_list) is the number of sprinklings with that path length
      
    return results

def δ_vs_t(D, N_values, height, min_freq, individual_sprinklings=False):
    """
    Plots r vs. t for all points of all sprinklings with the same path length.
    min_freq = which path lengths to plot based on how many sprinklings produced that path length
    if min_freq == False, then only the most frequent path length is plotted for each N

    """
    all_results = []
    rng = np.random.default_rng()
    ## This is the loop in which we gather all the results 
    # sequentially for each N by building sprinklings, then calling data_rs_ts_all(D, height, N, sprinklings)

    for N in N_values:
        sprinklings = sprinklings_D_H_N(D, height, N)
        all_results.extend(data_rs_ts_all(D, height, N, sprinklings))

    # Convert results to a DataFrame for easy plotting
    df_results = pd.DataFrame(all_results, columns=["D", "Height", "N", "num_levels", "rs", "ts", "thetas", "phis", "num_sprinklings"])
    print(df_results)

    # plt.figure(figsize=(18, 12), dpi=300)

    for N in N_values:
        subset = df_results[df_results["N"] == N]  # Filter rows for current N
        max_chunks = subset["num_sprinklings"].max()
        
        ## Filter the subset to only include rows where num_sprinklings is the maximum for this N
        if individual_sprinklings:
            subset_filtered = subset[rng.integers(0, len(subset), 1)]  # select one random sprinkling of varying path length but same N
        else:
            if min_freq == False: 
                subset_filtered = subset[subset["num_sprinklings"] == max_chunks]
            else:
                subset_filtered = subset[subset["num_sprinklings"] >= min_freq]  # Further filter by min_freq

        # fixed ranges
        t_min, t_max = 0, height
        x_min, x_max = -0.50, 0.50
        y_min, y_max = -0.50, 0.50
        
        for _, row in subset_filtered.iterrows():
            rs = np.array(row["rs"])  # Convert to NumPy array
            thetas = np.array(row["thetas"])  # Convert to NumPy array
            phis = np.array(row["phis"])  # Convert to NumPy array

            xs = rs * np.cos(thetas)
            ys = rs * np.sin(thetas)
            ts = np.array(row["ts"])  # Convert to NumPy array

            num_levels = row["num_levels"]
            freq = row["num_sprinklings"]
            
            # Create a unique folder for this process
            parent_path = write_path()
            folder = "rs vs t"
            subfolder = f"D {D} - Height {height}"
            folder_path = join(parent_path, folder, subfolder)
            file_name = f"D{D}_N{N}_H{height}_levels{num_levels}_freq{freq}"
            file_path = join(folder_path, file_name)
            os.makedirs(folder_path, exist_ok=True)

            if D == 2:
                # Create the plot for r vs t, with t on the y-axis
                plt.figure(figsize=(20, 20), dpi=600)
                plt.scatter(xs, ts, s=1, marker=".")
                
                # fixed axis + equal scale
                plt.ylim(t_min, t_max)
                plt.xlim(x_min, x_max)
                plt.axis("equal")
                
                # gridlines
                # Keep x-ticks default
                plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.7)

                # Set y-ticks at twice the usual density
                yticks = np.linspace(t_min, t_max, 11)  # 11 ticks → spacing of 0.05 if range=0.5
                plt.yticks(yticks)

                plt.xlabel('x coordinate of the elements')
                plt.ylabel('t')
                plt.title(f'For N = {N}, D = {D}, H = {height} -- x vs. t for all sprinklings with path length {num_levels+1} with frequency {freq}')
                plt.savefig(f'{file_path}.png')
                plt.close()
            
            if D == 3: 
                fig = go.Figure(
                    data=go.Scatter3d(x=xs, y=ys, z=ts, mode='markers', marker=dict(size=2))
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title='x coordinate of the elements',
                        yaxis_title='y coordinate of the elements',
                        zaxis_title='t',
                        xaxis=dict(range=[x_min, x_max]),
                        yaxis=dict(range=[y_min, y_max]),
                        zaxis=dict(range=[t_min, t_max]),
                        aspectratio=dict(x=1, y=1, z=1)
                    ),
                    title=f'For N = {N}, D = {D}, H = {height} -- x,y vs. t for all sprinklings with path length {num_levels+1} with frequency {freq}'
                )

                fig.write_html(f'{file_path}.html', auto_open=False)

            if D == 4:
                raise NotImplementedError("3D plot for D=4 is not implemented.")    
            
            
    return

########## Visualizing one sprinkling's r vs t with a path on top and delta's dotted lines ##########
from functools import lru_cache
def num_paths(sprinkling_groups, target, source):
    """
    Counts the number of paths from target to source in a directed graph represented by element_dict.

    Can be changed to return all paths by storing them in a list instead of just counting by changing num_of_paths to a list called paths...
    """
    # create element_dict where key = element, value = list of its preds
    # Choose the most frequent path length
    max_freq = 0
    chosen_path_length = None
    for path_length, sprinklings in sprinkling_groups.items():
        if len(sprinklings) > max_freq:
            max_freq = len(sprinklings)
            chosen_path_length = path_length

    # Take the first sprinkling of that path length
    chosen_sprinkling = sprinkling_groups[chosen_path_length][0]  

    # create list of preds for each element in the chosen sprinkling's LMCs
    element_dict = {e: preds for e, _, _, _, _, _, preds in chosen_sprinkling}
    # def paths_helper(element_dict, target, source, path=None, visited=None, num_of_paths=None):
    #     if path is None:
    #         path = []
    #     if visited is None:
    #         visited = set()
    #     if num_of_paths is None:
    #         num_of_paths = 0

    #     path.append(target)
    #     visited.add(target)

    #     if target == source:
    #         num_of_paths += 1
    #     else:
    #         for node in element_dict.get(target, []):
    #             if node not in visited:
    #                 num_of_paths = paths_helper(element_dict, node, source, path, visited, num_of_paths)
    #     path.pop()
    #     visited.remove(target)
    #     return num_of_paths
    # num_of_paths = paths_helper(element_dict, target, source)
    def count_paths(element_dict, target, source):
        @lru_cache(maxsize=None)
        def f(u):
           if u == source:
               return 1
           return sum(f(v) for v in element_dict.get(u, ()))
        return f(target)
    num_of_paths = count_paths(element_dict, target, source)
    return num_of_paths

def wrapper_num_paths(Ds, heights, Ns):
    num_of_paths_dict = {}
    for N in Ns:
        sprinkling_groups = sprinklings_D_H_N(Ds[0], heights[0], N)
        #print(sprinkling_groups)
        num_of_paths_dict[N] = num_paths(sprinkling_groups, N + 1, 0)
    num_of_paths = np.array([num_of_paths_dict[N] for N in Ns], dtype=np.float64)
    
    X = Ns
    Y = np.log(num_of_paths)
    slope, intercept = np.polyfit(X, Y, 1)
    b = slope
    A = np.exp(intercept)
    print(f"Estimated b for p = A*exp(b*N): {b:.4f}")
    print(f"Estimated A for p = A*exp(b*N): {A:.4f}")

    plt.figure(figsize=(12, 8), dpi=400)
    plt.scatter(X, num_of_paths, label='log of data', color='blue')
    plt.plot(Ns, A * np.exp(b * np.array(Ns)), color='red', label='Fitted Curve: $A e^{-bN}$')    
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Log(Number of paths from s to t)')
    plt.title('Exponential fit: p = A exp(b*N)')
    plt.legend()
    plt.savefig(f'Number_of_paths_vs_N_D{Ds[0]}_H{heights[0]}.png')
    plt.grid(True)

def visualize_sprinkling_with_path(D, height, N_inside, sprinkling_groups):
    """
    Docstring for visualize_sprinkling_with_path
    Chooses the most frequent path length's first sprinkling and plots r vs t for all elements in the LMCs, and overlays one path from s to t on top of it in red.

    :param D: Description
    :param height: Description
    :param N: Description
    :param sprinkling_groups: Description
    """
    
    # Choose the most frequent path length
    max_freq = 0
    chosen_path_length = None
    for path_length, sprinklings in sprinkling_groups.items():
        if len(sprinklings) > max_freq:
            max_freq = len(sprinklings)
            chosen_path_length = path_length

    # Take the first sprinkling of that path length
    chosen_sprinkling = sprinkling_groups[chosen_path_length][0]  

    # create list of preds for each element in the chosen sprinkling's LMCs
    list_labels_preds = {e: preds for e, _, _, _, _, _, preds in chosen_sprinkling}

    target_element = N_inside + 1  # Assuming N_inside excludes s and t, and the last element is the target
    source_element = 0  # Assuming the first element is the source
    
    # extract rs, ts, thetas
    rs = np.array([r for _, _, _, r, _, _, _ in chosen_sprinkling])
    ts = np.array([t for _, _, t, _, _, _, _ in chosen_sprinkling])
    thetas = np.array([theta for _, _, _, _, theta, _, _ in chosen_sprinkling])

    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)    

    # Plot x, y vs t for all elements in the LMCs
    plt.figure(figsize=(20, 20), dpi=600)
    if D == 2:
        plt.scatter(xs, ts, s=10, color="blue", marker="o")
        plt.axis("equal")
        plt.xlabel('x coordinate of the elements')
        plt.ylabel('t')
        plt.title(f'For N = {N_inside}, D = {D}, H = {height} -- x vs. t for a sprinkling with path length {chosen_path_length} with frequency {max_freq}')
        
    elif D == 3:
        raise NotImplementedError("2D plot for D=3 is not implemented.")
    elif D == 4:
        raise NotImplementedError("3D plot for D=4 is not implemented.")    

    ## Now plot the path on top of the sprinkling  
    if D == 2:
        # Extract the path points using preds
        path_points = []
        current_element = target_element

        # from this point onwards, I can either find all the paths from target to source
        # or just trace back one path using the first element in the preds list for each element.
        # Here, I will just trace back one path.
        while current_element != source_element:
            path_points.append(current_element)
            current_preds = list_labels_preds[current_element]
            if not current_preds:
                break  # No predecessors found, exit the loop
            current_element = current_preds[0]  # Take the first predecessor
            #print(f"Current preds: {current_preds}, Current element: {current_element}")
        path_points.append(source_element)
        path_points.reverse()  # Reverse to get path from source to target

        # Extract x, y, t for path points by first creating a coord dict that later allows us to extract coordinates based on path order
        coord = {e: (t, r, theta) for e, _, t, r, theta, _, _ in chosen_sprinkling}

        path_rs = np.array([coord[e][1] for e in path_points])
        path_ts = np.array([coord[e][0] for e in path_points])
        path_thetas = np.array([coord[e][2] for e in path_points])
        
        path_xs = path_rs * np.cos(path_thetas)

        # Plot path on top of sprinkling
        plt.plot(path_xs, path_ts, '-o', markersize=10, color='red', label='Path')
        plt.savefig(f'visualize_sprinkling_D{D}_N{N_inside}_H{height}_pathlength{chosen_path_length}.png')
        plt.close()
    else:
        raise NotImplementedError("Path plotting for D>2 is not implemented.")
    return list_labels_preds

######### Before understanding how height affects delta ########
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
    
    Ds = [4]
    heights = [1, 10]

    inside_Ns = [100,  1309,  2618, 3927,  5236,  6545,  7854,  9163, 10472, 13090, 15708, 18326, 20944, 26180, 39270, 52360, 74048, 104720, 148096, 209440]
    # inside_Ns = [18326, 20944, 26180, 39270, 52360, 74048, 104720, 148096, 209440, 296193, 418880, 592368, 1184736]
    # inside_Ns = [1675470]
    
    # For Height 1 and 10
    # inside_Ns = [  654,  1309,  2618,  3927,  5236,  6545,  7854,  9163, 10472, 13090, 15708, 18326, 20944]
    
    #### Delta vs N plot ####
    # folder_path = read_path(Ds[0], heights[0])

    # with error bars: 
    # all_deltas_list = all_deltas(Ds[0], heights[0], inside_Ns, True, folder_path)
    # deltabyH_vs_N(all_deltas_list, Ds[0], heights[0], inside_Ns)

    #### bounds on beta and A ####
    # num_of_sprinklings(Ds, heights, inside_Ns, folder_path)
    # all_deltas_D2_H1 = all_deltas(Ds[0], heights[0], inside_Ns, True, folder_path)
    # bound_beta, bound_A = bootstrap_bounds(all_deltas_D2_H1, inside_Ns, 2000)
    # print(f"bound_beta = {bound_beta}")
    # print(f"bound_A = {bound_A}")

    # df_delta_data = all_delta_data(mid_layers)
    
    #### Delta vs t plot ####
    # min_freq = False
    # individual_sprinklings = True
    # δ_vs_t(Ds[0], inside_Ns, heights[0], min_freq, individual_sprinklings)
    
    #### individual sprnkling with path ####
    # for N_inside in inside_Ns:
    #     sprinkling_groups = sprinklings_D_H_N(Ds[0], heights[0], N_inside)
    # #print(sprinkling_groups)
    #     visualize_sprinkling_with_path(Ds[0], heights[0], N_inside, sprinkling_groups)

    #### number of paths vs. N ####
    # wrapper_num_paths(Ds, heights, inside_Ns)
    
    df = all_delta_data(
        use_middle_third=True,              # True = middle third, False = all times
        Ds=Ds,
        heights=heights,
        N_values=inside_Ns,
        delta_weighting="equal_sprinkling"  # equal sprinkling weights
    )

    δbyH_vs_N(Ds, df, delta_definition=True, annotate_points=True)

    
    
