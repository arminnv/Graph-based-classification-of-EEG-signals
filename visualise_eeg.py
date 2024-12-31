import time

import matplotlib
import matplotlib.pyplot as plt
import mne
# For elimiating warnings
import mne.viz
from warnings import simplefilter
# ignore all future warnings
import scipy
import numpy as np
from channels import electrodes
matplotlib.use('TkAgg')
ch_names = list(electrodes.keys())
simplefilter(action='ignore', category=FutureWarning)
fs = 256
threshold = 0.0
t_start = int(0*fs)    #6
t_end = int(10*fs)      #10
print(ch_names)

#Load epoched data
data = scipy.io.loadmat(f'raw_data/A.mat')['data']
data = data[0, 0]
trial = data['trial'][0, 0].T[0].astype(int)

x = np.array(data['X'][0, 0])
x = np.array([x[trial[i] + t_start: trial[i] + t_end] for i in range(len(trial))])
x = np.mean(x, axis=0)
print(x.shape)
# Read the EEG epochs:
#epochs = mne.read_epochs(data_file + '.fif', verbose='error')
eeg_data = x
#raw = mne.io.RawArray(x, mne.create_info(ch_names, fs))

# Assuming your EEG data is stored in a NumPy matrix called 'eeg_data'
# and the sampling frequency is 'sfreq'
#rawData = genfromtxt('EEG-S1.csv', delimiter=',')
#dataCut = np.delete(rawData,np.s_[0:2],axis=1)
#dataCut = np.transpose(dataCut)
rawData = x.T
#plt.plot(rawData[])
#plt.show()
# Creating Info
info = mne.create_info(
        ch_names = ch_names,
        sfreq    =  fs,
        ch_types=  'eeg')
raw = mne.io.RawArray(rawData,info,verbose='info')

# Plotting raw data together with events
raw.plot(start=0, duration=20, clipping=None, scalings='auto',
       block=True)

"""
epochs = epochs['FP', 'FN', 'FU']
print('Percentage of Pleasant familiar events : ', np.around(len(epochs['FP'])/len(epochs), decimals=2))
print('Percentage of Neutral familiar events : ',np.around(len(epochs['FN'])/len(epochs), decimals=2))
print('Percentage of Unpleasant familiar : ', np.around(len(epochs['FU'])/len(epochs), decimals=2))
"""

#time.sleep(20)
#raw.average().plot()
"""
from matplotlib import pyplot as plt


fp = epochs['FP']
fn = epochs['FN']
fu = epochs['FU']
ch = 18
conditions = ['FP', 'FN', 'FU']

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Time Instances')
ax.set_ylabel('Volt')

ax.plot(fp.average().data[ch, :], color='blue', label='Familiar Pleasant')
ax.plot(fn.average().data[ch, :], color='red', label='Familiar Neutral')
ax.plot(fu.average().data[ch, :], color='green', label='Familiar Unpleasant')

legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
plt.title('ERP of different conditions')
plt.show()

times = np.arange(0, 1, 0.1)
epochs.average().plot_topomap(times, ch_type='eeg')

import pylab, seaborn as sns
from scipy.stats import ttest_rel, sem


def plot_conditions(data, times, plot_title):
    sns.set(style="white")
    ColorsL = np.array(([228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0])) / 256
    col_axes = np.array((82, 82, 82)) / 256

    al = 0.2
    fig = plt.figure(num=None, figsize=(4, 2), dpi=150)

    epochs_mean = np.mean(data, axis=0)
    epochs_std = sem(data, axis=0) / 2

    plt.plot(times, epochs_mean, color=ColorsL[0], linewidth=2)
    plt.fill_between(times, epochs_mean, epochs_mean + epochs_std, color=ColorsL[0], interpolate=True, alpha=al)
    plt.fill_between(times, epochs_mean, epochs_mean - epochs_std, color=ColorsL[0], interpolate=True, alpha=al)
    plt.ylabel('Mean ERP')
    plt.xlabel('Times')
    plt.title(plot_title)
"""
