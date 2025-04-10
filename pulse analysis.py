import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal
import pandas as pd
import scikit_posthocs as sp
import itertools

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
duration_values = list(duration.values())
IPI_values = list(IPI.values())
fly_labels = list(carriers.keys())

filtered_IPI = {}
for fly_name, values in IPI.items():
    values = values[~np.isnan(values)]
    filtered = values[values <= 80]
    if len(filtered) > 0:
        # print(f"{fly_name}: {len(filtered)} values kept (max={np.max(filtered):.2f})")
        filtered_IPI[fly_name] = filtered

##############################
# ## scatter plot and box plot, normality test, significance test
def analyze_and_plot(feature_dict, title, ylabel):
    ## remove NaN
    feature_dict = {k: v[~np.isnan(v)] for k, v in feature_dict.items() if len(v) > 0}
    
    ## change the file name into fly1, fy2, .... 
    original_labels = list(feature_dict.keys())
    fly_labels = [f"Fly {i+1}" for i in range(len(original_labels))]
    label_map = dict(zip(original_labels, fly_labels))

    ## normality test
    for k, v in feature_dict.items():
        stat, p = shapiro(v)
        print(f"{label_map[k]}: p = {p:.4f} {'(not normally distributed)' if p < 0.05 else '(normally distributed)'}")

    ## Kruskal-Wallis test
    data_groups = list(feature_dict.values())
    stat, p = kruskal(*data_groups)
    if p >= 0.05:
        print("Significant difference: No")
    else:
        print("Significant difference: Yes")

    ## Posthoc Dunn-Bonferroni test
    all_data = []
    groups = []
    for name, vals in feature_dict.items():
        all_data.extend(vals)
        groups.extend([label_map[name]] * len(vals))  
    df = pd.DataFrame({ylabel: all_data, 'group': groups})
    pvals_df = sp.posthoc_dunn(df, val_col=ylabel, group_col='group', p_adjust='bonferroni')

    ## plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    values_list = list(feature_dict.values())

    for i, values in enumerate(values_list):
        x = np.full_like(values, i, dtype=int)
        jitter_x = x + np.random.normal(loc=0, scale=0.1, size=len(values))
        ax.scatter(jitter_x, values, alpha=0.1, s=10)

    ax.boxplot(values_list, positions=range(len(fly_labels)), widths=0.6, patch_artist=False, showfliers=False)
    ax.set_xticks(range(len(fly_labels)))
    ax.set_xticklabels(fly_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.show

    ## Save posthoc test result to CSV
    save_dir = 'D:/Learning materials/Oldenburg/2nd semester/Neuroethology and Neurogenetics_Insect Models/data/'
    os.makedirs(save_dir, exist_ok=True)
    result_filename = os.path.join(save_dir, f'posthoc_{ylabel}.csv')
    pvals_df.to_csv(result_filename)
    ## Also save significance markers (***, **, *, -) version
    stars_df = pvals_df.copy()
    for i in range(stars_df.shape[0]):
        for j in range(stars_df.shape[1]):
            pval = stars_df.iat[i, j]
            if i == j or pd.isna(pval):
                stars_df.iat[i, j] = ''
            elif pval < 0.001:
                stars_df.iat[i, j] = '***'
            elif pval < 0.01:
                stars_df.iat[i, j] = '**'
            elif pval < 0.05:
                stars_df.iat[i, j] = '*'
            else:
                stars_df.iat[i, j] = '-'

    stars_filename = os.path.join(save_dir, f'posthoc_{ylabel}_stars.csv')
    stars_df.to_csv(stars_filename)
    
    ## Save label mapping (Fly n -> filename)
    mapping_df = pd.DataFrame({
        'Fly Label': fly_labels,
        'Original Filename': original_labels
    })
    mapping_filename = os.path.join(save_dir, f'label_mapping_{ylabel}.csv')
    mapping_df.to_csv(mapping_filename, index=False)

analyze_and_plot(filtered_IPI, "IPI comparison across flies (≤80ms)", "IPI")
analyze_and_plot(duration, "Duration comparison across flies", "Duration")
analyze_and_plot(carriers, "Carrier frequency comparison across flies", "Carrier")
