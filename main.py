
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
    for name in files:
        if name.endswith((".rar")):

            rar_path = os.path.join(root, name)
            print(rar_path)
            extract_path = os.path.join(root, name.replace('.rar', ''))
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
import pandas
import numpy as np

g_indices = list(range(1,9))
gestures = ["Acceleration" + str(idx) for idx in g_indices]
gestures



#%% Preprocessing

G = 9.80665 # acceleration of gravity



#%% Feature extraction


#%% Training


#%% Test
 