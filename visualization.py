import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Select a gesture and plot the acceleration for all axes for a single repetition
def visualize_axes(my_data, ges, idx):
    fig = plt.figure(figsize=(10,3))
    for i in range(3):
        plt.subplot(1,3,i+1)
        y = my_data[ges][idx][:,i]
        x = range(len(y))
        plt.scatter(x=x, y=y, s=1)
    
    plt.show()


# Visualize the data on a selected plane
def visualize_plane(my_data, ges, idx, ax1, ax2):
    fig = plt.figure(figsize=(6,4))
    x = my_data[ges][idx][:,ax1]
    y = -1.0 * my_data[ges][idx][:,ax2]
    plt.scatter(x=x, y=y, c=np.arange(len(my_data[ges][idx])), cmap='viridis')


# Interactive 3D visualization
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