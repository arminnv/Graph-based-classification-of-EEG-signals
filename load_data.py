import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import electrode_names
from sklearn.utils import shuffle

subjects = ['E']  # "['A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L']
sessions = [1]  # , 2]
fs = 256
t_start = int(4.5 * fs)  # 6
t_end = int(7.5 * fs)  # 10
exclusions = []

# Delta Theta Alpha Beta Gamma
fbands = {'Delta': (0.5, 4),
          'Theta': (4, 8),
          'Alpha': (8, 12),
          'Beta': (12, 35),
          'Gamma': (35, 100)}

X = []
y = []

for subject in subjects:
    for session in sessions:
        data = scipy.io.loadmat(f'raw_data/{subject}.mat')['data']
        data = data[0, session - 1]
        trial = data['trial'][0, 0].T[0].astype(int)

        x = data['X'][0, 0]
        x = np.delete(x, exclusions, axis=1)
        print(x.shape)
        x = np.array([x[trial[i] + t_start: trial[i] + t_end] for i in range(len(trial))])
        X.append(x)
        y.append(data['y'][0, 0].T[0].astype(int))
        print(len(trial))

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
print(y.shape)

# A, y = shuffle(A, y)

"""
20-30 features
MRMR
sequential forward selection
E F subject
add graph features
remove channels
4.5-7.5

"""
