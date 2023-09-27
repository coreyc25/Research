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

# Import data into Pandas DataFrame
geo_data = pd.read_csv("mostTraversedLatLong.csv", usecols = ['Lat', 'Long'])
geo_data = np.array(geo_data)

# SOM algorithm
clusters = 3
features = len(geo_data[0])
som = SOM(m = clusters, dim = features)

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