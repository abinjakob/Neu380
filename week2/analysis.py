#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:00:42 2025


WEEK 02 Analysis of Pulse Songs and Sine Songfs for Transgenically modified Flies
---------------------------------------------------------------------------------

The experiment involved modified male flies where the PC2 neurons are activated. There
were three conditions (i) before PC2 activation (Before Light), (ii) During PC2 activation
(During Light), (iii) after PC2 activation (After Light)
with optogenetics.

This script performs:
    - loading the data 
    - plot the raw time domain data 
    - test the normality of the mean of each flies for each condition
    - test the Wilcoxon test to test the median of each condition
    - plots the box plot with signifcance marked


@author: Abin Jacob, Yifan Pan, Laman
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""


#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, wilcoxon

#%% load data 

data = np.load(r'/Users/abinjacob/Documents/04. Uni Oldenburg/06 SEM/neu380 Neurogenetics/DATA/Analysis Data/annotations/pC2_chrimson (1).npz')
time = data['time']
pulse = data['pulse']
sine = data['sine']

#%% plot pulse and sine data

# pulse data 
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
for fly_p in range(len(pulse)):
    plt.plot(time, pulse[fly_p], linewidth = 0.5, alpha=0.5)
plt.plot(time, np.mean(pulse, axis=0), 'k')
plt.axvspan(0, 5, color = 'r', alpha = 0.05)
plt.ylabel('Probability')
plt.xlabel('Time (sec)')
plt.title('Pulse Songs')
plt.ylim(0,0.5)


# sine data 
plt.subplot(1,2,2)
for fly_p in range(len(sine)):
    plt.plot(time, sine[fly_p], linewidth = 0.5, alpha=0.5)
plt.plot(time, np.mean(sine, axis = 0), 'k')
plt.axvspan(0, 5, color = 'r', alpha = 0.05)
plt.ylabel('Probability')
plt.xlabel('Time (sec)')
plt.title('Sine Songs')
plt.ylim(0,0.5)

# save figure
plt.savefig('/Users/abinjacob/Documents/04. Uni Oldenburg/06 SEM/neu380 Neurogenetics/DATA/Analysis Data/Week_02/rawplot.png')


#%% preparing data tests

# indices of before, during and after periods
t_bef = [t for t in time if t <= 0]
t_dur = [t for t in time if t > 0 and t <= 5]
t_aft = [t for t in time if t > 5]


# --- pulse songs 
pulse_bef, pulse_dur, pulse_aft  = [], [], []

# loop over flies 
for f in range(len(pulse)):
    bef, dur, aft = [], [], []
    # loop over data points
    for i in range(len(time)):
        if time[i] < 0:
            bef.append(pulse[f][i])
        if time[i] >= 0 and time[i] <= 5:
            dur.append(pulse[f][i])
        if time[i] > 5:
            aft.append(pulse[f][i])
    
    # create mat for each flies 
    pulse_bef.append(bef)
    pulse_dur.append(dur)
    pulse_aft.append(aft)


# --- sine songs 
sine_bef, sine_dur, sine_aft  = [], [], []

# loop over flies 
for f in range(len(sine)):
    bef, dur, aft = [], [], []
    # loop over data points
    for i in range(len(time)):
        if time[i] < 0:
            bef.append(sine[f][i])
        if time[i] >= 0 and time[i] <= 5:
            dur.append(sine[f][i])
        if time[i] > 5:
            aft.append(sine[f][i])
    
    # create mat for each flies 
    sine_bef.append(bef)
    sine_dur.append(dur)
    sine_aft.append(aft)
    
# sine songs mean for each flies
pulse_bef_mean = np.mean(pulse_bef, axis=1)
pulse_dur_mean = np.mean(pulse_dur, axis=1)
pulse_aft_mean = np.mean(pulse_aft, axis=1)
# pulse songs mean for each flies
sine_bef_mean  = np.mean(sine_bef, axis=1)
sine_dur_mean  = np.mean(sine_dur, axis=1)
sine_aft_mean  = np.mean(sine_aft, axis=1)

#%% Normality Test on mean of each flies

# pulse song
k, pb = shapiro(pulse_bef_mean)
k, pd = shapiro(pulse_dur_mean)
k, pa = shapiro(pulse_aft_mean)
print(f'Pulse Song: {pb}, {pd}, {pa}')
print('\n')
# sine song
k, pb = shapiro(sine_bef_mean)
k, pd = shapiro(sine_dur_mean)
k, pa = shapiro(sine_aft_mean)
print(f'Pulse Song: {pb}, {pd}, {pa}')


#%% Wilcoxon Tests on mean of each flies

# pulse song
k, pbd_p = wilcoxon(pulse_bef_mean, pulse_dur_mean)
k, pba_p = wilcoxon(pulse_bef_mean, pulse_aft_mean)
print(f'Pulse Song: {pbd_p}, {pba_p}')
print('\n')
# sine song
k, pbd_s = wilcoxon(sine_bef_mean, sine_dur_mean)
k, pba_s = wilcoxon(sine_bef_mean, sine_aft_mean)
print(f'Sine Song: {pbd_s}, {pba_s}')

#%% Plotting boxplots for different conditions with significant vals

# function to calculate significance values
def annotatePvals(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

# plot box plots 
plt.figure(figsize=(10, 5))

# -- sine songs
# concatenating pulse data conditions 
pulse_data = [pulse_bef_mean, pulse_dur_mean, pulse_aft_mean]
# caluclate significance for pulse
pbd_p_annotation = annotatePvals(pbd_p) 
pba_p_annotation = annotatePvals(pba_p) 
# plotting box plots for pulse songs
plt.subplot(1,2,1)
plt.boxplot(pulse_data, labels = ['Before light', 'During light', 'After light'])
plt.title('Pulse Songs')
# annotate significance values
y_max = max([max(i) for i in pulse_data])
y_min = min([min(i) for i in pulse_data])
h = (y_max - y_min) * 0.1 
x1, x2 = 1, 2
y = y_max + h
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, pbd_p_annotation, ha='center', va='bottom', fontsize=12)
x1, x2 = 1, 3
y = y + h*1.5
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, pba_p_annotation, ha='center', va='bottom', fontsize=12)
plt.ylim(0,0.35)

# -- sine songs
# concatenating sine data conditions 
sine_data = [sine_bef_mean, sine_dur_mean, sine_aft_mean]
# caluclate significance for sine
pbd_s_annotation = annotatePvals(pbd_s) 
pba_s_annotation = annotatePvals(pba_s) 
# plotting box plots for sine songs
plt.subplot(1,2,2)
plt.boxplot(sine_data, labels = ['Before light', 'During light', 'After light'])
plt.title('Sine Songs')
# annotate significance values
y_max = max([max(i) for i in sine_data])
y_min = min([min(i) for i in sine_data])
h = (y_max - y_min) * 0.1 
x1, x2 = 1, 2
y = y_max + h
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, pbd_s_annotation, ha='center', va='bottom', fontsize=12)
x1, x2 = 1, 3
y = y + h*1.5
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, pba_s_annotation, ha='center', va='bottom', fontsize=12)
plt.ylim(0,0.35)

# save figure
plt.savefig('/Users/abinjacob/Documents/04. Uni Oldenburg/06 SEM/neu380 Neurogenetics/DATA/Analysis Data/Week_02/boxplot.png')

