import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import shapiro, f_oneway, kruskal
import pandas as pd

path = 'D:/Learning materials/Oldenburg/2nd semester/Neuroethology and Neurogenetics_Insect Models/data/pulse results/annotations/'
pulse = glob.glob(os.path.join(path, '*.npz'))

carriers = {}
duration = {}
IPI = {}
for file_path in pulse:
    data = np.load(file_path)
    # print(data)
    file_name = os.path.basename(file_path)
    if 'carrier' in data:
        carriers[file_name] = data['carrier']
    if 'duration' in data:
        duration[file_name] = data['duration']
    if 'interval' in data:
        IPI[file_name] = data['interval']
carrier_values = list(carriers.values())
fly_labels = list(carriers.keys())
duration_values = list(duration.values())
fly_labels = list(duration.keys())
IPI_values = list(IPI.values())
fly_labels = list(IPI.keys())

flycolor = [ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.4, 0.7], [0.5, 0.1, 0.1], [0.7, 0.2, 0.1],
            [0.7, 0.4, 0.2], [0.1, 0.3, 0.2], [0.3, 0.6, 0.3], [0.4, 0.7, 0.3], [0.2, 0.1, 0.5],
            [0.4, 0.4, 0.6], [0.5, 0.5, 0.7], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6] ]
## Carrier frequency plot ########
labels = []
plt.figure(figsize=(10, 6))
for i1, (fly_name, values) in enumerate(carriers.items()):
    x1 = np.full_like(values, i1, dtype=int)
    jitter_x1 = x1 + np.random.normal(loc=0, scale=0.1, size=len(values))
    plt.scatter(jitter_x1, values, color = flycolor[i1], alpha=0.1, s=10)
    labels.append(f'Fly {i1+1}')
carrier_values_list = list(carriers.values())
positions = range(len(fly_labels))
plt.boxplot(carrier_values_list, positions=positions, widths=0.6, patch_artist=True)

plt.xticks(carrier_values_list, labels)
plt.xlabel('Flies')
plt.ylabel('Carrier frequency')
plt.title('Carrier frequency comparison across individual flies')

plt.show()

## Duration plot ########
plt.figure(figsize=(10, 6))
for i2, (fly_name, values) in enumerate(duration.items()):
    x2 = np.full_like(values, i2, dtype=int)
    jitter_x2 = x2 + np.random.normal(loc=0, scale=0.1, size=len(values))
    plt.scatter(jitter_x2, values, alpha=0.1, label=fly_name, s=10)
plt.xlabel('Flies')
plt.ylabel('Duration')
plt.title('Duration comparison across individual flies')
# plt.show()

## IPI plot ########
plt.figure(figsize=(10, 6))
filtered_IPI = {}
for fly_name, values in IPI.items():
    values = values[~np.isnan(values)]
    filtered = values[values <= 80]
    if len(filtered) > 0:
        # print(f"{fly_name}: {len(filtered)} values kept (max={np.max(filtered):.2f})")
        filtered_IPI[fly_name] = filtered

for i3, (fly_name, values) in enumerate(filtered_IPI.items()):
    x3 = np.full_like(values, i3, dtype=int)
    jitter_x3 = x3 + np.random.normal(loc=0, scale=0.1, size=len(values))
    plt.scatter(jitter_x3, values, alpha=0.1, label=fly_name, s=10)
plt.xlabel('Flies')
plt.ylabel('IPI')
plt.title('IPI comparison across individual flies')
# plt.show()

## Normality test ########################
def normal(feature):
    all_normal = True
    # print("Normality test (Shapiro-Wilk):")
    for fly_name, values in feature.items():
        stat, p = shapiro(values)
        if p <= 0.05:
            # print(f"{fly_name}: not noramlly distributed (p = {p:.3f})")
            all_normal = False

    if all_normal:
        print("all normally distributed")
    else:
        print("not normally distributed")

normal(carriers)
normal(duration)
normal(filtered_IPI)

## Significance comparison ##################

def kruskal_test(feature, feature_name):
    data_groups = list(feature.values())
    
    stat, p = kruskal(*data_groups)
    print(f"\nKruskal-Wallis H ({feature_name}):")
    print(f"  p value: {p:.3f}")
    alpha = 0.05
    if p > alpha:
        print("no significant difference")
    else:
        print("there is significant difference")

kruskal_test(carriers, "Carrier")
kruskal_test(duration, "Duration")
kruskal_test(filtered_IPI, "IPI") 

## Posthoc test ################

# def 
