import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation

# Load the data
data = pd.read_csv("stateEvolution.csv", delim_whitespace=True, header=None)

# Number of rows and columns in the grid
N = 100

# Function to update the plot
def update(frame):
    grid = data.iloc[frame].values.reshape(N, N)
    mat.set_array(grid)  # Use set_array instead of set_data
    magnetization = np.sum(grid)
    ax.set_title(f"Frame: {frame}, Magnetization: {magnetization:.2f}")
    fig.canvas.draw_idle()  # Ensure the canvas is redrawn
    return [mat]

# Create a figure and axis
fig, ax = plt.subplots()
grid = data.iloc[0].values.reshape(N, N)
mat = ax.matshow(grid, cmap='gray')  # Use 'gray' for black and white color scheme

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=len(data), interval=50, blit=True)

# Display the animation
plt.show()