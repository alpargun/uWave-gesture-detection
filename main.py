
#%% Import statements



#%%

import os
from rarfile import RarFile

# Extract .rar files
# Important: Mac users should make sure to have "unrar" or "unar" in the PATH
# Works on MacOS after installing "unar" with "brew install unar"

DATA_ROOT = "data/uWaveGestureLibrary"

# Find all .rar files, extract them into separate folders and remove the .rar files
for root, dirs, files in os.walk(DATA_ROOT):
    print('root: ', root, ' dirs: ', dirs)
    for file in files:
        if file.endswith((".rar")):

            rar_path = os.path.join(root, file)
            print(rar_path)
            extract_path = os.path.join(root, file.replace('.rar', ''))
            os.mkdir(extract_path)

            with RarFile(rar_path) as rf: 
                rf.extractall(path=extract_path)
            
            os.remove(rar_path)

#TODO 
# check if directory exists, check if empty etc

#%% Load data

# Top level:
# $userIndex is the index of the participant from 1 to 8, and $dayIndex is the index
# of the day from 1 to 7.

# Inside each .rar file:
# .txt files recording the time series of acceleration of each
# gesture. The .txt files are named as [somePrefix]$gestureIndex-$repeatIndex.txt, where
# $gestureIndex is the index of the gesture as in the 8-gesture vocabulary, and
# $repeatIndex is the index of the repetition of the same gesture pattern from 1 to 10.

# Summary
# 8 users. 7 days
# 8 gestures. 10 repetitions for each gesture by each user.
# For each gesture, 3 columns: x-, y-, z-axis accelerations (unit is G)
# 

import pandas as pd
import numpy as np
import pickle

g_indices = list(range(1,9))
GESTURES = ["Acceleration" + str(idx) for idx in g_indices]

# Create a dictionary where the key is gesture index and the values are series for each gesture
dict_gestures = {gesture: [] for gesture in GESTURES}

# file name should include $gestureIndex. Skip other files
for root, dirs, files in os.walk(DATA_ROOT):
    print('root: ', root, ' dirs: ', dirs)
    for file in files:
        
        for gesture in GESTURES:
            if gesture in file and file.endswith('.txt'):
                print('file:', file)
                data = pd.read_csv(os.path.join(root, file), delimiter=" ", header=None)
                dict_gestures[gesture].append(data.to_numpy())

# Save data using pickle
path_save = 'data/combined-data/'
os.mkdir(path_save)
with open(os.path.join(path_save, 'gestures.pkl'), 'wb') as f:
    pickle.dump(dict_gestures, f)

#TODO
# Try extracting and loading in the same loop for efficiency

#%% Preprocessing

G = 9.80665 # acceleration of gravity



#%% Feature extraction


#%% Training


#%% Test
 