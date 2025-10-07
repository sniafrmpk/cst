#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:28:11 2024

@author: naumanibrahim
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def r_analyzer(N):
    if N != 65000 and N != 80000:
    # Load the Excel file
        df = pd.read_excel(f"spherical_coordinates_N{N}_l_0.xlsx")
        # Assuming the 'r' values are in a column called 'r'
        # Drop rows where 'r' is NaN (if there's an empty row)
        r = df.dropna(subset=['r'])['r']
    else:
        df = pd.read_csv(f"spherical_coordinates_N{N}_l_0.csv", skiprows=1)
        r = df.iloc[:, 1]
    #   r = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Convert to numeric
    # Split the dataframe into chunks by finding rows where r is 0
    chunks = []
    current_chunk = []

    for value in r:
        if value == 0:
            if current_chunk:  # Add the current chunk to the list
                chunks.append(current_chunk)
                current_chunk = []  # Reset the current chunk
        else:
            current_chunk.append(value)

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    # Calculate the mean of each chunk
    means = [np.mean(chunk) for chunk in chunks]
   
    # Calculate the mean of the means
    mean_of_means = np.mean(means)

    # Calculate the standard error of the means
    std_error = np.std(means)

    print(f"Mean of means for N = {N}:", mean_of_means)
    print(f"Standard error of the means for N = {N}:", std_error)

    return mean_of_means, std_error

means_of_means = []
list_of_std_errors = []

for N in [5000, 10000, 20000, 30000, 40000, 50000, 65000, 80000]:
    mean_of_means, std_error = r_analyzer(N)
    means_of_means.append(mean_of_means)
    list_of_std_errors.append(std_error)

Ns = [5000, 10000, 20000, 30000, 40000, 50000, 65000, 80000]

plt.plot(Ns, means_of_means)
plt.show()












