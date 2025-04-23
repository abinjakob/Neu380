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
from scipy.stats import shapiro, mannwhitneyu

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


#%% plot histograms 

limit_columns = ['male_rotational_speed', 'female_rotational_speed']

columns_to_plot = [
    'male_rotational_speed', 'male_velocity_magnitude', 'male_velocity_forward',
    'male_velocity_lateral', 'male_acceleration_mag', 'female_rotational_speed',
    'female_velocity_magnitude', 'female_velocity_forward', 'female_velocity_lateral',
    'female_acceleration_mag', 'distance', 'male_relative_angle',
    'male_relative_orientation', 'male_relative_velocity_mag',
    'female_relative_angle', 'female_relative_orientation', 'female_relative_velocity_mag'
]

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

for coulumn in columns_to_plot:
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

#%% Mann–Whitney U test

for coulumn in columns_to_plot:

    stat, p = mannwhitneyu(mean_individual_sine[column].dropna(), mean_individual_pulse[column].dropna(), alternative='two-sided')
    if p < 0.001:
        psign = '**'
    elif p < 0.05:
        psign = '**'
    else:
        psign = 'Nope'
        
    print(f'{column}= p-value: {psign}')


