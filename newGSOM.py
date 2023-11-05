"""
Example using simple SOM to cluster sklearn's Iris dataset.

NOTE: This example uses sklearn and matplotlib, but neither is required to use
the som (sklearn is used here just for the data and matplotlib just to plot
the output).

@author: Riley Smith
Created: 2-1-2021
"""
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
        #self.weights = np.random.rand(size, size, features)
        self.weights = np.random.rand(size, size, features)  # Initialize as an empty array
        self.iteration = 0
    
    """
    Params:
        winner: is the winning neuron found from training (had the minimum distance)
        k: is the current iteration value being called
    """
    def adaptWeights(self, winner, p):
      # Calculate the distance between each neuron and the winning neuron
      distances = self.weights - p.reshape(1, 1, -1)  # Reshape p for broadcasting

      # Find the neurons within the specified radius
      neighborhood = []
      for row_idx in range(self.size):
          for col_idx in range(self.size):
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
          new_delta += np.sum(np.abs(delta))

      print(new_delta)
      return new_delta

    def growNode(self, p):
        p = p.reshape(1, -1)  # Reshape p to match the number of features
        if self.weights is None:
            self.weights = p  # Initialize self.weights with the first data point
        else:
            self.weights = np.concatenate((self.weights, p), axis=0)





def find_bmus(trained_weights, data):
    """
    Find Best-Matching Units (BMUs) for each data point in the dataset.

    Parameters:
    - trained_weights (numpy.ndarray): The trained weights of the Self-Organizing Map.
    - data (numpy.ndarray): The dataset for which to find BMUs.

    Returns:
    - bmu_indices (numpy.ndarray): An array containing the BMU indices for each data point.
    """

    # Check if the dimensions of the trained_weights and data are compatible
    if trained_weights.shape[-1] != data.shape[-1]:
        raise ValueError("The last dimension of trained_weights and data must match.")

    # Reshape the trained_weights array to a 2D array where each row corresponds to a neuron
    trained_weights_2d = trained_weights.reshape(-1, trained_weights.shape[-1])

    # Initialize an array to store BMU indices for each data point
    bmu_indices = np.zeros((data.shape[0],), dtype=int)

    # Iterate over each data point in the dataset
    for i, data_point in enumerate(data):
        # Calculate the Euclidean distance between the data point and all neurons
        # in the trained_weights_2d array
        distances = np.linalg.norm(trained_weights_2d - data_point, axis=1)

        # Find the index of the neuron with the minimum distance (the BMU)
        bmu_index = np.argmin(distances)

        # Store the BMU index for this data point
        bmu_indices[i] = bmu_index

    return bmu_indices

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
tolerance = 0.5

while True:  # Change to an infinite loop
    weights_delta = 0  # Initialize weights_delta for each iteration
    for p in geo_data:
        winner, error = FindWinner(neurons.weights, p)
        weights_delta += neurons.adaptWeights(winner, p)  # Accumulate weights_delta
        if error >= GT:
            neurons.growNode(p)
        neurons.iteration += 1

    # Check if the accumulated weights_delta is less than the tolerance
    if weights_delta < tolerance:
        break  # Exit the loop if the accumulated delta is less than the tolerance
