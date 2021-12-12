
#%% Import statements



#%% Extract rar files into corresponding folders

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
    #print('root: ', root, ' dirs: ', dirs)
    for file in files:
        
        for gesture in GESTURES:
            if gesture in file and file.endswith('.txt'):
                #print('file:', file)
                data = pd.read_csv(os.path.join(root, file), delimiter=" ", header=None)
                dict_gestures[gesture].append(data.to_numpy())

# Save data using pickle
path_save = 'data/combined-data/'
os.mkdir(path_save)
with open(os.path.join(path_save, 'gestures.pkl'), 'wb') as f:
    pickle.dump(dict_gestures, f)

#TODO
# Try extracting and loading in the same loop for efficiency


#%% Load pickled data

with open('data/combined-data/gestures.pkl', 'rb') as f:
    data = pickle.load(f)

data


#%%
# Integrate
import scipy.integrate

# A function to do numerical integration
def integrate(my_data):
    data_integ = {gesture: [] for gesture in my_data}
    for key in my_data:
        for i in range(len(my_data[key])):
            integrated = scipy.integrate.cumulative_trapezoid(my_data[key][i], axis=0)
            data_integ[key].append(integrated)
    return data_integ

data_vel = integrate(data) # Velocity data
data_pos = integrate(data_vel) # Position data


#%% Visualize each axis separately
# Select a gesture and plot the acceleration for all axes for a single repetition

import matplotlib.pyplot as plt

def visualize(my_data, ges, idx):
    fig = plt.figure(figsize=(10,3))
    for i in range(3):
        plt.subplot(1,3,i+1)
        y = my_data[GESTURES[1]][1][:,i]
        x = range(len(y))
        plt.scatter(x=x, y=y, s=1)
    
    plt.show()

gesture = GESTURES[0]
idx_ges = 0
visualize(data, gesture, idx_ges) # Acceleration
visualize(data_vel, gesture, idx_ges) # Velocity
visualize(data_pos, gesture, idx_ges) # Position


#%% Visualize the data on a selected plane

def visualize_plane(my_data, ges, idx, ax1, ax2):
    fig = plt.figure(figsize=(6,4))
    x = my_data[ges][idx][:,ax1]
    y = -1.0 * my_data[ges][idx][:,ax2]
    plt.scatter(x=x, y=y, c=np.arange(len(my_data[ges][idx])), cmap='viridis')

gesture = GESTURES[0]
idx_ges = 0
ax1, ax2 = 0, 1
visualize_plane(data_pos, gesture, idx_ges, ax1, ax2)

#%% Interactive 3D Plotting

import plotly.express as px

def visualize_3d(my_data, ges, idx):
    time_ax = range(len(my_data[ges][idx])) # To have a color scheme for time

    fig = px.scatter_3d(x=my_data[ges][idx][:,0],
                        y=my_data[ges][idx][:,1],
                        z=my_data[ges][idx][:,2],
                        color=time_ax
                        )
    # Default parameters which are used when `layout.scene.camera` is not provided
    scene=dict(
            camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.25, y=1.25, z=1.25)), #the default values are 1.25, 1.25, 1.25
            xaxis=dict(),
            yaxis=dict(),
            zaxis=dict(),
            aspectmode='data', #this string can be 'data', 'cube', 'auto', 'manual'
            #a custom aspectratio is defined as follows:
            aspectratio=dict(x=1, y=1, z=1)
            )

    fig.update_layout(scene=scene)
    fig.show()

gesture = GESTURES[0]
idx_ges = 0
visualize_3d(data_vel, gesture, idx_ges)

#%%





#%% Preprocessing

# Normalization

# Numerical integration


#%% Feature Extraction/ reduction

# Polynomial fit



#%% Training


#%% Test
 