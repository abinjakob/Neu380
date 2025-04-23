#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 10:19:22 2025


WEEK 03 
---------------------------------------------------------------------------------




@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""

#%% libraries 
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from scipy.stats import shapiro, wilcoxon
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

#%% load data 

# file to load 
file2load = 'song_and_tracking.csv'
# rooth path 
rootpath  = r'/Users/abinjacob/Documents/04. Uni Oldenburg/06 SEM/neu380 Neurogenetics/DATA' 
filename = op.join(rootpath, file2load)
# load the xlsx data 
df = pd.read_csv(filename)

# getting sine data 
sine_data = df[df['pulse'] == 1]
# calculating mean sine data per individual per column
mean_individual_sine = sine_data.groupby('file').median().reset_index()

# getting pulse data 
pulse_data = df[df['pulse'] == 0]
# calculating mean pulse data per individual per column
mean_individual_pulse = pulse_data.groupby('file').median().reset_index()


limit_columns = ['male_rotational_speed', 'female_rotational_speed']

columns_to_plot = [
    'male_rotational_speed', 'male_velocity_magnitude', 'male_velocity_forward',
    'male_velocity_lateral', 'male_acceleration_mag', 'female_rotational_speed',
    'female_velocity_magnitude', 'female_velocity_forward', 'female_velocity_lateral',
    'female_acceleration_mag', 'distance', 'male_relative_angle',
    'male_relative_orientation', 'male_relative_velocity_mag',
    'female_relative_angle', 'female_relative_orientation', 'female_relative_velocity_mag'
]

#%% plot histograms 

for column in columns_to_plot:
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))

    data_sine = sine_data[column].dropna()
    axes.hist(data_sine, bins=100, edgecolor='black', color='blue', alpha=0.5, density=True, label= 'Sine')
    axes.set_title(f'Sine - {column}')
    axes.set_xlabel(column)
    axes.set_ylabel('Density')
    if column in limit_columns:
        axes.set_xlim(0, 45) 

    data_pulse = pulse_data[column].dropna()
    axes.hist(data_pulse, bins=100, edgecolor='black', color='red', alpha=0.5, density=True, label= 'Pulse')
    axes.set_title(f'Pulse - {column}')
    axes.set_xlabel(column)
    axes.set_ylabel('Density')
    if column in limit_columns:
        axes.set_xlim(0, 45)

    plt.tight_layout()
    plt.legend()
    plt.show()
    
#%% noirmality test 


for column in columns_to_plot:
    k, ps = shapiro(sine_data[column].dropna())
    k, pp = shapiro(pulse_data[column].dropna())
    if ps > 0.05:
        s_norm = 'Normal'
    else:
        s_norm = 'Nope'
    if pp > 0.05:
        p_norm = 'Normal'
    else:
        p_norm = 'Nope'
        
    print(f'{column}= sine: {s_norm}, pulse: {p_norm}')

#%% Calculating Wilcoxon test 


fig, axes = plt.subplots(3, 6)
axes = axes.flatten() 
# calculate wilcoxon test
for idx, column in enumerate(columns_to_plot):
    
    # extract the values for the current column
    sine_vals  = mean_individual_sine[column].dropna()
    pulse_vals = mean_individual_pulse[column].dropna()

    # performing wilcoxon test 
    stat, p = wilcoxon(sine_vals, pulse_vals, alternative='two-sided')
    if p < 0.001:
        psign = '***'
    if p < 0.01:
        psign = '**'
    elif p < 0.05:
        psign = '*'
    else:
        psign = 'ns'
    
    # plotting the data for the current column
    ax = axes[idx]
    for i in range(10):
        ax.plot([0,1], [sine_vals[i], pulse_vals[i]], marker='o', alpha=0.7)
    ax.set_xticklabels(['Sine', 'Pulse'], fontsize=10)
    ax.set_title(f'{column} ({psign})', fontsize=10)
    plt.tight_layout()

#  removing the last axes 
fig.delaxes(axes[17])



#%% calculating the UMAP


# get the feature vector
X = df[columns_to_plot]
labels = df['pulse']
X_clean = X.dropna()
X_scaled = StandardScaler().fit_transform(X_clean)
labels_clean = labels[X_clean.index]


reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
# reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_scaled)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_clean, cmap='coolwarm', s=5, alpha=0.6)
plt.title('UMAP Projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()
