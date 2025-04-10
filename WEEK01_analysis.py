#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:00:42 2025

WEEK 01 Analysis of Pulse Songs for Wild Type 
----------------------------------------------

This script performs:
    - loads the feature files 
    - pad or truncate features
    - create feature vector and lables
    - noprmalise the data using StandardScalar()
    - Performs PCA (n_components=2)
    - Create scatter plots for PC1 and PC2


@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""

#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import glob
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% load the data

# fetch filenmaes in the rootpath 
rootpath = r'/Users/abinjacob/Documents/04. Uni Oldenburg/06 SEM/neu380 Neurogenetics/DATA/Analysis Data/annotations'
pulsefiles = glob.glob(op.join(rootpath, '*.npz'))

#  params
# threshold to remove ipi between pulse  trains
ipi_thresh = 100
# make feature vec equal length for PCA
fixed_len = 800

# init store feature values
duration = {}
carrier = {}
ipi = {}
# init feature and label vector for PCA
features = []
labels = []

# colors for flies
flycolor = [ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.4, 0.7], [0.5, 0.1, 0.1], [0.7, 0.2, 0.1],
            [0.7, 0.4, 0.2], [0.1, 0.3, 0.2], [0.3, 0.6, 0.3], [0.4, 0.7, 0.3], [0.2, 0.1, 0.5],
            [0.4, 0.4, 0.6], [0.5, 0.5, 0.7], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6] ]

# load files
for idx, file in enumerate(pulsefiles):
    data = np.load(file)
    duration[idx] = data['duration']
    carrier[idx] = data['carrier']
    ipi[idx] = data['interval'][data['interval'] > 35]
    

#%% create the feature vector

# function to pad or truncate arrays
def pad_or_truncate(arr, length):
    if len(arr) >= length:
        return arr[:length]
    else:
        return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
    
# create feat and label vectors for PCA
for idx in duration.keys():
    dur = duration[idx]
    ipi_vals = ipi[idx]
    carr = carrier[idx]

    if len(dur) == 0 or len(ipi_vals) == 0 or len(carr) == 0:
        continue

    # Create 1D feature vector by concatenating
    f = np.concatenate([
        pad_or_truncate(dur, fixed_len),
        pad_or_truncate(ipi_vals, fixed_len),
        pad_or_truncate(carr, fixed_len)
    ])
    features.append(f)
    labels.append(f'Fly {idx+1}')  # or use fly ID here

X = np.array(features)

# impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
# standardize values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

#%% PCA computing
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#%% plotting PCA
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Label'] = labels

sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', s=100, palette= flycolor)
plt.title('PCA of Pulse Song Features')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
