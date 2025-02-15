import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation

# Load the data
data = pd.read_csv("Results/stateEvolution.csv", delim_whitespace=True, header=None)

# Number of rows and columns in the grid
N = 100

# Function to update the plot
def update(frame):
    grid = data.iloc[frame].values.reshape(N, N)
    mat.set_array(grid)  
    magnetization = np.sum(grid)
    ax.set_title(f"Frame: {frame}, Magnetization: {magnetization:.2f}")
    fig.canvas.draw_idle()  
    return [mat]

# Create a figure and axis
fig, ax = plt.subplots()
grid = data.iloc[0].values.reshape(N, N)
mat = ax.matshow(grid, cmap='gray')

# Add a legend
import matplotlib.patches as mpatches
spin_1_patch = mpatches.Patch(color='white', label='Spin 1')
spin_neg1_patch = mpatches.Patch(color='black', label='Spin -1')
plt.legend(handles=[spin_1_patch, spin_neg1_patch], loc='upper right')

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=len(data), interval=50, blit=True)

# Save the animation as a GIF
save_as_gif = False
if save_as_gif:
    ani.save("Results/stateEvolution.gif", writer='Pillow', fps=20)

# Display the animation
plt.show()