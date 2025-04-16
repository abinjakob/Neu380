import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal, ttest_ind
import scipy

data = np.load('D:/Learning materials/Oldenburg/2nd semester/Neuroethology and Neurogenetics_Insect Models/data/week2/pC2_chrimson (1).npz')
time = data['time']
pulse = data['pulse']
sine = data['sine']

#############################################

for fly_p in range(len(pulse)):
    plt.plot(time, pulse[fly_p], linewidth = 0.2)
plt.plot(time, np.mean(pulse, axis=0), 'k')
plt.axvspan(0, 5, color = 'r', alpha = 0.05)
plt.ylabel('Probability of pulse song')
plt.xlabel('Time (sec)')
plt.show()

t_bef = [t for t in time if t <= 0]
t_dur = [t for t in time if t > 0 and t <= 5]
t_aft = [t for t in time if t > 5]

all_bef = []
all_dur = []
all_aft = []

for f in range(len(pulse)):
    for i in range(len(time)):
        if time[i] < 0:
            all_bef.append(pulse[f][i])
        if time[i] >= 0 and time[i] <= 5:
            all_dur.append(pulse[f][i])
        if time[i] > 5:
            all_aft.append(pulse[f][i])
    
print(len(all_bef),len(all_aft))
k, pb = shapiro(all_bef)
k, pd = shapiro(all_dur)
k, pa = shapiro(all_aft)

######################################

p = [all_bef, all_dur, all_aft]
# normally distributed


k, pbd = ttest_ind(all_bef, all_dur)
# k, pba = ttest_ind(all_bef, all_aft)
print(f'{pbd:.10f}')
plt.figure()
plt.boxplot(p, labels = ['Before light', 'During light', 'After light'], showfliers=False)
plt.ylabel('Probability of pulse song')
plt.show()

##########################

for fly_p in range(len(sine)):
    plt.plot(time, sine[fly_p], linewidth = 0.2)
plt.plot(time, np.mean(sine, axis = 0), 'k')
plt.axvspan(0, 5, color = 'r', alpha = 0.05)
plt.ylabel('Probability of sine song')
plt.xlabel('Time (sec)')
# plt.show()

t_bef = [t for t in time if t <= 0]
t_dur = [t for t in time if t > 0 and t <= 5]
t_aft = [t for t in time if t > 5]

all_bef = []
all_dur = []
all_aft = []

for f in range(len(sine)):
    for i in range(len(time)):
        if time[i] <= 0:
            all_bef.append(sine[f][i])
        if time[i] > 0 and time[i] <= 5:
            all_dur.append(sine[f][i])
        if time[i] > 5:
            all_aft.append(sine[f][i])
     
k, pb = shapiro(all_bef)
k, pd = shapiro(all_dur)
k, pa = shapiro(all_aft)
print(f'{pb:.5f}, {pd:.5f}, {pa:.5f}')

###########################################3

for fly_p in range(len(sine)):
    plt.plot(time, sine[fly_p], linewidth = 0.2)
plt.plot(time, np.mean(sine, axis = 0), 'k')
plt.axvspan(0, 5, color = 'r', alpha = 0.05)
plt.ylabel('Probability of sine song')
plt.xlabel('Time (sec)')
# plt.show()

t_bef = [t for t in time if t <= 0]
t_dur = [t for t in time if t > 0 and t <= 5]
t_aft = [t for t in time if t > 5]

all_bef = []
all_dur = []
all_aft = []

for f in range(len(sine)):
    bef = []
    dur = []
    aft = []
    for i in range(len(time)):
        if time[i] <= 0:
            bef.append(sine[f][i])
        if time[i] > 0 and time[i] <= 5:
            dur.append(sine[f][i])
        if time[i] > 5:
            aft.append(sine[f][i])
    all_bef.append(bef)
    all_dur.append(dur)
    all_aft.append(aft)
m_bef = np.mean(all_bef, axis=1)
m_dur = np.mean(all_dur, axis = 1)
m_aft = np.mean(all_aft, axis = 1)
print(m_bef)
k, pb = shapiro(m_bef)
k, pd = shapiro(m_dur)
k, pa = shapiro(m_aft)
print(f'{pb:.5f}, {pd:.5f}, {pa:.5f}')
## bef, aft: 

#######################################

p = [all_bef, all_dur, all_aft]
# normally distributed

k, pbd = ttest_ind(all_bef, all_dur)
k, pba = ttest_ind(all_bef, all_aft)
print(f'{pbd:.5f}', pba)
plt.figure()
plt.boxplot(p, labels = ['Before light', 'During light', 'After light'], showfliers=False)
plt.ylabel('Probability of pulse song')


plt.show()
