import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
magnetization_data = pd.read_csv("Results/MagnetizationHistory.csv")
energy_data = pd.read_csv("Results/EnergyHistory.csv")

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the magnetization data
ax1.plot(magnetization_data.index, magnetization_data.values, 'r.', label='Magnetization')
ax1.set_xlabel("Index")
ax1.set_ylabel("Magnetization")
ax1.set_title("Magnetization History Plot")
ax1.legend()

# Plot the energy data
ax2.plot(energy_data.index, energy_data.values, 'b.', label='Energy')
ax2.set_xlabel("Index")
ax2.set_ylabel("Energy")
ax2.set_title("Energy History Plot")
ax2.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()