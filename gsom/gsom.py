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
from neuron import Neurons

def visualize(data):
    latitude = data[:, :, 0]
    longitude = data[:, :, 1]

    # Create subplots for latitude and longitude
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Create a heatmap for latitude
    ax1.imshow(latitude, cmap='viridis')
    ax1.set_title('Latitude')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Create a heatmap for longitude
    ax2.imshow(longitude, cmap='viridis')
    ax2.set_title('Longitude')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add color bars
    cbar1 = plt.colorbar(ax1.imshow(latitude, cmap='viridis'), ax=ax1)
    cbar2 = plt.colorbar(ax2.imshow(longitude, cmap='viridis'), ax=ax2)

    plt.tight_layout()
    plt.show()


# Spread factor
sf = 1

# Growth threshold
GT = -2 * math.log(sf)

def find_bmus(trained_weights, data):

    # Check if the dimensions of the trained_weights and data are compatible
    if trained_weights.shape[-1] != data.shape[-1]:
        raise ValueError("The last dimension of trained_weights and data must match.")

    # Reshape the trained_weights array to a 2D array where each row corresponds to a neuron
    trained_weights_2d = trained_weights.reshape(-1, trained_weights.shape[-1])

    # Initialize an array to store BMU indices for each data point
    bmu_indices = np.zeros((data.shape[0],), dtype=int)
    threshold = 100
    # Iterate over each data point in the dataset
    for i, data_point in enumerate(data):
        # Calculate the Euclidean distance between the data point and all neurons
        # in the trained_weights_2d array
        distances = np.linalg.norm(trained_weights_2d - data_point, axis = 1)
        adjusted_dist = []
        for val in distances:
            if val > threshold:
                adjusted_dist.append(val)

        # Filters the distances based on the threshold to exclude distances greater than the threshold
        valid_distances = distances[distances <= threshold]
        bmu_index = np.argmin(valid_distances)

        # Find the index of the neuron with the minimum distance (the BMU)
        #bmu_index = np.argmin(adjusted_dist)

        # Store the BMU index for this data point
        bmu_indices[i] = bmu_index

    return bmu_indices

# Parses the CSV file and finds min and max values
def find_min_max_values(geo_data):

    # Finds min and max values for each column
    minValues = np.min(geo_data, axis=0)
    

    maxValues = np.max(geo_data, axis=0)

    return minValues, maxValues


if __name__ == '__main__':

    # Import data into Pandas DataFrame
    geo_data = pd.read_csv("mostTraversedLatLong.csv", usecols = ['Lat', 'Long'])
    geo_data = np.array(geo_data)

    

    neuron_size = 10
    features = 2
    neighborhood_radius = 5
    learning_rate = 0.02
    min_array, max_array = find_min_max_values(geo_data)
    neurons = Neurons(neuron_size, features, neighborhood_radius, learning_rate, min_array, max_array)

    

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
            weights_delta += neurons.adaptWeights(winner, p)
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
    rounded_weights = []
    # for i in neurons.weights:
    #     for j in i:
    #         for k in j:
    #             rounded_weights.append(round(k, 2))
    # print(rounded_weights)

    # Plot the results
    # x = geo_data[:,0]
    # y = geo_data[:,1]
    # colors = ['red', 'green', 'blue']

    # plt.scatter(x, y, c=clusters, cmap=ListedColormap(colors))
    # plt.show()
    # visualize(neurons.weights)

    
    #clusters = find_bmus(neurons.weights, geo_data)
    
    clusters = find_bmus(neurons.weights, geo_data)
    print(clusters)