"""
Example using simple SOM to cluster sklearn's Iris dataset.

NOTE: This example uses sklearn and matplotlib, but neither is required to use
the som (sklearn is used here just for the data and matplotlib just to plot
the output).

@author: Riley Smith
Created: 2-1-2021
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn_som.som import SOM
from sklearn import datasets
import pandas as pd
import numpy as np
import math

# Import data into Pandas DataFrame
geo_data = pd.read_csv("mostTraversedLatLong.csv", usecols = ['Lat', 'Long'])
geo_data = np.array(geo_data)

# SOM algorithm
clusters = 3
n = len(geo_data)
k = len(geo_data[0])
som = SOM(m = clusters, dim = k)

# Spread factor
sf = 1

# neighborhood radius, initially large
radius = 1

# Growth threshold
GT = -2 * math.log(sf)

def FindWinner(neurons, p):
    # Convert the dataset element 'd' into a NumPy array for easy computation
    p = np.array(p)
    
    # Initialize variables to keep track of the minimum error and the corresponding winning neuron
    min_error = float('inf')
    winning_neuron = None
    
    # Iterate through the neurons in the set and find the neuron with the smallest error
    for neuron in neurons:
        # Calculate the Euclidean distance between the dataset element 'd' and the neuron
        error = np.linalg.norm(neuron - p)
        
        # If the error is smaller than the current minimum, update the minimum error and the winning neuron
        if error < min_error:
            min_error = error
            winning_neuron = neuron
    
    return winning_neuron, min_error

def AdaptWeights(winner, p, radius):
    new_delta = 0
    return new_delta

weights_delta = float('inf')
tolerance = 0.1

D = geo_data
N = "neurons network"
neurons = np.random.rand(n,k)
distances = []

while weights_delta > tolerance:
    weights_delta = 0
    for p in D:
        winner, error = FindWinner(N, p)
        weights_delta = AdaptWeights(winner, p, radius)
        if error >= GT:
            GrowNode(N, p)
        N.iteration += 1



# Algorithm 2 Smoothing
r_large = r * 2
for i in range(50):
    for p in D:
        winner, error = FindWinner(N, p)
        AdaptWeights(winner, p, r_large)

r_small = r * 0.5
for i in range(50):
    for p in D:
        winner, error = FindWinner(N, p)
        AdaptWeights(winner, p, r_small)

# Everything below here is for visualization

# Fit algorithm to the data
som.fit(geo_data)

# Assign each datapoint to its predicted cluster
predictions = som.predict(geo_data)

# Plot the results
x = geo_data[:,0]
y = geo_data[:,1]
colors = ['red', 'green', 'blue']

# Visualization
plt.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
plt.show()