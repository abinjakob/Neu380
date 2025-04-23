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
from scipy.stats import gaussian_kde

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


columns4umap = [
    'male_velocity_magnitude', 'male_velocity_forward','male_velocity_lateral', 'male_acceleration_mag', 'female_rotational_speed',
    'female_velocity_magnitude', 'female_velocity_forward','female_acceleration_mag', 'distance', 
     'male_relative_velocity_mag','female_relative_velocity_mag'
]

# get the feature vector
X = df[columns4umap]
labels = df['pulse']
X_clean = X.dropna()
X_scaled = StandardScaler().fit_transform(X_clean)
labels_clean = labels[X_clean.index]


reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_scaled)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_clean, cmap='coolwarm', s=5, alpha=0.6)
plt.title('UMAP Projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()


#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Prepare UMAP embeddings for pulse and sine
embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
embedding_df['pulse'] = labels_clean.values

pulse_points = embedding_df[embedding_df['pulse'] == 0][['UMAP1', 'UMAP2']].values.T
sine_points = embedding_df[embedding_df['pulse'] == 1][['UMAP1', 'UMAP2']].values.T

# Create grid
xmin, xmax = embedding[:, 0].min(), embedding[:, 0].max()
ymin, ymax = embedding[:, 1].min(), embedding[:, 1].max()
Xgrid, Ygrid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])

# KDE for each class
kde_pulse = gaussian_kde(pulse_points)(positions).reshape(Xgrid.shape)
kde_sine = gaussian_kde(sine_points)(positions).reshape(Xgrid.shape)

# Difference in densities
kde_diff = kde_pulse - kde_sine

# Plot
plt.figure()
plt.subplot(1,3,1)
plt.contourf(Xgrid, Ygrid, kde_sine, cmap='coolwarm', levels=20)
plt.subplot(1,3,2)
plt.contourf(Xgrid, Ygrid, kde_pulse, cmap='coolwarm', levels=20)
plt.subplot(1,3,3)
plt.contourf(Xgrid, Ygrid, kde_diff, cmap='coolwarm', levels=20)
plt.colorbar(label='Pulse - Sine Density')
plt.title('Density Difference in UMAP Space (Pulse - Sine)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()


#%% plotting KDE


# UMAP pulse and sine points
umap_sine = embedding[labels_clean == 1]
umap_pulse = embedding[labels_clean == 0]


# creating a mesh grid over the UMAP space
xmin, xmax = embedding[:,0].min(), embedding[:,0].max()
ymin, ymax = embedding[:,1].min(), embedding[:,1].max()
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
grid_coords = np.vstack([xx.ravel(), yy.ravel()])

# KDEs over 2D space
kde_pulse = gaussian_kde(umap_pulse.T)
kde_sine = gaussian_kde(umap_sine.T)
zz_pulse = kde_pulse(grid_coords).reshape(xx.shape)
zz_sine = kde_sine(grid_coords).reshape(xx.shape)
zz_diff = zz_pulse - zz_sine

# plottiing
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# pulse density
axs[0].imshow(zz_pulse, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='Reds', alpha=0.7)
axs[0].set_title("Pulse")
# sine density
axs[1].imshow(zz_sine, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='Purples', alpha=0.7)
axs[1].set_title("Sine")
# difference (Pulse - Sine)
diff_plot = axs[2].imshow(zz_diff, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='bwr', alpha=0.8)
axs[2].set_title("Pulse - Sine")
fig.colorbar(diff_plot, ax=axs[2], shrink=0.8, label='Density Difference')

plt.tight_layout()
plt.show()
