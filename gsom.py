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

# In terms of implementation, what is the best way to represent a neighborhood of nodes
class Neurons:
    def __init__(self, size, features, radius, lr):
        self.size = size
        self.count = size * size
        self.radius = radius
        self.lr = lr
        self.weights = np.random.rand(size, size, features)
        self.iteration = 0
    
    """
    Params:
        winner: is the winning neuron found from training (had the minimum distance)
        k: is the current iteration value being called
    """
    def adaptWeights(self, winner, p):
        # Calculate the distance between each neuron and the winning neuron
        distances = self.weights - p
        
        # Find the neurons within the specified radius
        neighborhood = []
        for row_idx in range(len(self.weights)):
            for col_idx in range(len(self.weights[0])):
                cur_neuron = self.weights[row_idx][col_idx]
                distance = np.linalg.norm(winner - cur_neuron)
                if distance <= self.radius:
                    neighborhood.append((row_idx, col_idx))

        learning_factor = (self.iteration + 1) / self.count
        self.lr = math.pow(0.02, learning_factor)

        # Update the weights of neurons within the radius
        new_delta = 0
        for row_idx, col_idx in neighborhood:
            delta = self.lr * distances[row_idx][col_idx]
            self.weights[row_idx][col_idx] += delta
            new_delta += abs(delta[0]) + abs(delta[1])

        return new_delta
    
    def growNode(self, p):
        self.weights = np.append(self.weights, p)

if __name__ == '__main__':

    # Import data into Pandas DataFrame
    geo_data = pd.read_csv("mostTraversedLatLong.csv", usecols = ['Lat', 'Long'])
    geo_data = np.array(geo_data)

    neuron_size = 10
    features = 2
    neighborhood_radius = 2
    learning_rate = 0.02
    neurons = Neurons(neuron_size, features, neighborhood_radius, learning_rate)

    # Spread factor
    sf = 1

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

    # Algorithm 1 Learn and Grow
    weights_delta = float('inf')
    tolerance = 0.5

    while weights_delta > tolerance:
        weights_delta = 0
        for p in geo_data:
            winner, error = FindWinner(neurons.weights, p)
            weights_delta = neurons.adaptWeights(winner, p)
            if error >= GT:
                neurons.growNode(p)
            neurons.iteration += 1

    # Algorithm 2 Smoothing
    neurons.radius = neighborhood_radius * 2
    for i in range(50):
        for p in geo_data:
            winner, error = FindWinner(neurons.weights, p)
            neurons.adaptWeights(winner, p)

    neurons.radius = neighborhood_radius * 0.5
    for i in range(50):
        for p in geo_data:
            winner, error = FindWinner(neurons.weights, p)
            neurons.adaptWeights(winner, p)

    print(neurons.weights)
    # # Everything below here is for visualization

    # # Fit algorithm to the data
    # som.fit(geo_data)

    # # Assign each datapoint to its predicted cluster
    # predictions = som.predict(geo_data)

    # # Plot the results
    # x = geo_data[:,0]
    # y = geo_data[:,1]
    # colors = ['red', 'green', 'blue']

    # # Visualization
    # plt.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
    # plt.show()