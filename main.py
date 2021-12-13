
#%% Import statements
from visualization import visualize_axes, visualize_plane, visualize_3d
import os
from rarfile import RarFile
import pandas as pd
import numpy as np
import pickle


# --------------------------------------------------------------------------------
# PREPARE DATA
#%% Extract rar files into corresponding folders

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
            #print(rar_path)
            extract_path = os.path.join(root, file.replace('.rar', ''))
            os.mkdir(extract_path)

            with RarFile(rar_path) as rf: 
                rf.extractall(path=extract_path)
            
            os.remove(rar_path)

#TODO 
# check if directory exists, check if empty etc

#%% Load data

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
# Check if pkl file exists

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

# ----------------------------------------------------------------------------------
# VISUALIZATION
#%% Visualize each axis separately

gesture = GESTURES[0]
idx_ges = 0
visualize_axes(data, gesture, idx_ges) # Acceleration
visualize_axes(data_vel, gesture, idx_ges) # Velocity
visualize_axes(data_pos, gesture, idx_ges) # Position


#%% Visualize the data on a selected plane

gesture = GESTURES[0]
idx_ges = 0
ax1, ax2 = 0, 1
visualize_plane(data_pos, gesture, idx_ges, ax1, ax2)

#%% Interactive 3D Plotting

gesture = GESTURES[0]
idx_ges = 0
visualize_3d(data_vel, gesture, idx_ges)


# ---------------------------------------------------------------------------------
# FEATURE EXTRACTION
#%% Visualize Polynomial fit
import matplotlib.pyplot as plt

deg_poly = 3

visualize_axes(data_vel, GESTURES[0], 0)

val = data_vel[GESTURES[0]][0]
fig = plt.figure(figsize=(10,3))

poly = np.polyfit(x=range(len(val)), y=val, deg=deg_poly)
print(poly)

for i in range(3):
    poly_new = np.polyval(poly[:,i], range(len(val[:,i])))
    plt.subplot(1,3,i+1)
    plt.scatter(x=range(len(poly_new)), y=poly_new, s=1)

plt.show()


#%% Apply Polynomial Fit to All Data

deg_poly = 3

my_data = data_vel

data_poly = {gesture: [] for gesture in my_data}
for key in my_data:
    #print('key:', key)
    for idx, val in enumerate(my_data[key]):
        if len(val) > 1:                
            poly = np.polyfit(x=range(len(val)), y=val, deg=deg_poly)
            #print('poly: ', poly)
            data_poly[key].append(poly.T.flatten())


#%% Prepare feature and label matrices
len_data = sum(map(len, data_poly.values()))
num_features = len(data_poly[GESTURES[0]][0])
X = np.zeros((len_data, num_features), dtype=float)
y = np.zeros(len_data, dtype=int)

count = 0
for gesture_idx, gesture in enumerate(GESTURES):
    for val in data_poly[gesture]:
        X[count,:] = val
        y[count] = gesture_idx
        count += 1

#%% Prepare train and test sets

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

#%%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=0)


#%% Training
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, verbose=1, multi_class='ovr', solver='lbfgs', max_iter=200)

clf.fit(X_train, y_train)


#%% Test
from sklearn.metrics import confusion_matrix

score_test = clf.score(X_test, y_test)
print("Test set accuracy: ", score_test)


#-------------------------------------------------------------------------------------------------------------------
# EVALUATE
# %% Visualize results
import seaborn as sns

# Confusion matrix
y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat

fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, fmt='g')

# %%
