import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv("Results/MagnetizationReplica.csv")

run = 0 #also equal to the value of the random seed used


# Plot the data
plt.plot(data.index, data[data.columns[run]], 'r.')
plt.xlabel("index")
plt.ylabel("M")
plt.title("History Plot (S=" + str(run) + ")")
plt.show()