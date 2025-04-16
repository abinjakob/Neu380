import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal, ttest_ind
import scipy

data = np.load('D:/Learning materials/Oldenburg/2nd semester/Neuroethology and Neurogenetics_Insect Models/data/week2/pC2_chrimson (1).npz')
time = data['time']
pulse = data['pulse']
sine = data['sine']

#############################################
### pulse song traces 
for fly_p in range(len(pulse)):
    plt.plot(time, pulse[fly_p], linewidth = 0.2)
plt.plot(time, np.mean(pulse, axis = 0), 'k')
plt.axvspan(0, 5, color = 'r', alpha = 0.05)
plt.ylabel('Probability of pulse song')
plt.xlabel('Time (sec)')
# plt.show()

t_bef = [t for t in time if t <= 0]
t_dur = [t for t in time if t > 0 and t <= 5]
t_aft = [t for t in time if t > 5]

all_bef = []
all_dur = []
all_aft = []

for f in range(len(pulse)):
    bef = []
    dur = []
    aft = []
    for i in range(len(time)):
        if time[i] <= 0:
            bef.append(pulse[f][i])
        if time[i] > 0 and time[i] <= 5:
            dur.append(pulse[f][i])
        if time[i] > 5:
            aft.append(pulse[f][i])
    all_bef.append(bef)
    all_dur.append(dur)
    all_aft.append(aft)
m_bef = np.mean(all_bef, axis=1)
m_dur = np.mean(all_dur, axis = 1)
m_aft = np.mean(all_aft, axis = 1)
print(len(m_bef))
k, pb = shapiro(m_bef)
k, pd = shapiro(m_dur)
k, pa = shapiro(m_aft)
print(f'{pb:.5f}, {pd:.5f}, {pa:.5f}')
## bef: not normal, dur, aft: normal

######################################
## mean pulse probability for each fly
pulse_phase = [m_bef, m_dur, m_aft]

k, pbd = wilcoxon(m_bef, m_dur)
k, pba = ttest_rel(m_bef, m_aft)
print(f'{pbd:.5f}', pba)

def pval_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

star_bd = pval_to_star(pbd)  # before vs during
star_ba = pval_to_star(pba)  # before vs after

plt.figure()
plt.boxplot(pulse_phase, labels = ['Before light', 'During light', 'After light'])
plt.ylabel('Probability of pulse song')

y_max = max([max(i) for i in pulse_phase])
y_min = min([min(i) for i in pulse_phase])
h = (y_max - y_min) * 0.1 

x1, x2 = 1, 2
y = y_max + h
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, star_bd, ha='center', va='bottom', fontsize=12)

x1, x2 = 1, 3
y = y + h*1.5
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, star_ba, ha='center', va='bottom', fontsize=12)

plt.show()

##########################
## sine song traces
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
print(len(m_bef))
k, pb = shapiro(m_bef)
k, pd = shapiro(m_dur)
k, pa = shapiro(m_aft)
print(f'{pb:.5f}, {pd:.5f}, {pa:.5f}')
## bef, aft: not normal, dur: normal

###########################################3

## sine song probability of each fly
sine_phase = [m_bef, m_dur, m_aft]

k, pbd = wilcoxon(m_bef, m_dur)
k, pba = ttest_rel(m_bef, m_aft)

star_bd = pval_to_star(pbd)  # before vs during
star_ba = pval_to_star(pba)  # before vs after

print(f'{pbd:.5f}', pba)
plt.figure()
plt.boxplot(sine_phase, labels = ['Before light', 'During light', 'After light'])
plt.ylabel('Probability of sine song')

y_max = max([max(i) for i in sine_phase])
y_min = min([min(i) for i in sine_phase])
h = (y_max - y_min) * 0.1 

x1, x2 = 1, 2
y = y_max + h
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, star_bd, ha='center', va='bottom', fontsize=12)

x1, x2 = 1, 3
y = y + h*1.5
plt.plot([x1, x1, x2, x2], [y, y+h/4, y+h/4, y], lw=1.2, c='k')
plt.text((x1+x2)/2, y+h/3, star_ba, ha='center', va='bottom', fontsize=12)

plt.show()
