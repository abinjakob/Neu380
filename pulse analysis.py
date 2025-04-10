import numpy as np
import os
import glob
import matplotlib.pyplot as plt

path = 'D:/Learning materials/Oldenburg/2nd semester/Neuroethology and Neurogenetics_Insect Models/data/pulse results/annotations/'
pulse = glob.glob(os.path.join(path, '*.npz'))

# carriers = {}
# for file_path in pulse:
#     data = np.load(file_path)
#     file_name = os.path.basename(file_path)
#     print(data)
#     if 'carrier' in data:
#         carriers[file_name] = data['carrier']
# carrier_values = list(carriers.values())
# fly_labels = list(carriers.keys())

# for i, (fly_name, values) in enumerate(carriers.items()):
#     x = np.full_like(values, i, dtype=int)
#     plt.scatter(x, values, alpha=0.7, label=fly_name, s=10)
# plt.xlabel('Flies')
# plt.ylabel('Carrier frequency')
# plt.title('Carrier frequency comparison across individual flies')
# plt.tight_layout()
# plt.show()

duration = {}
for file_path in pulse:
    data = np.load(file_path)
    file_name = os.path.basename(file_path)
    
    if 'carrier' in data:
        duration[file_name] = data['carrier']
carrier_values = list(duration.values())
fly_labels = list(duration.keys())

for i, (fly_name, values) in enumerate(duration.items()):
    x = np.full_like(values, i, dtype=int)
    plt.scatter(x, values, alpha=0.7, label=fly_name, s=10)
plt.xlabel('Flies')
plt.ylabel('Duration')
plt.title('Duration comparison across individual flies')
plt.tight_layout()
plt.show()