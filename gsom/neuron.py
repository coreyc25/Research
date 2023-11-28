import math
import numpy as np

# In terms of implementation, what is the best way to represent a neighborhood of nodes
class Neurons:
    def __init__(self, size, features, radius, lr, min_values, max_values):
        self.size = size
        self.count = size * size
        self.radius = radius
        self.lr = lr
        random_weights = np.random.rand(size, size, features)
        self.weights = (max_values - min_values) * random_weights + min_values
        self.iteration = 0
    
    """
    Params:
        p: an element from the dataset
    """
    def FindWinner(self, p):
        # Convert the dataset element 'd' into a NumPy array for easy computation
        p = np.array(p)
        
        # Initialize variables to keep track of the minimum error and the corresponding winning neuron
        min_error = float('inf')
        winning_neuron = None
        
        # Iterate through the neurons in the set and find the neuron with the smallest error
        for neuron in self.weights:
            # Calculate the Euclidean distance between the dataset element 'd' and the neuron
            error = np.linalg.norm(neuron - p)
            
            # If the error is smaller than the current minimum, update the minimum error and the winning neuron
            if error < min_error:
                min_error = error
                winning_neuron = neuron
        
        return winning_neuron, min_error
    
    """
    Params:
        winner: is the winning neuron found from training (had the minimum distance)
        k: is the current iteration value being called
    """
    def adaptWeights(self, winner, p):
        # Calculate the distance between each neuron and the winning neuron
        distances = self.weights - p
        
        # Find the neurons within the specified radius
        learning_factor = (self.iteration + 1) / self.count
        self.lr = math.pow(0.02, learning_factor)

        delta = 0
        # Update the weights of neurons within the radius
        for row_idx in range(len(self.weights)):
            for col_idx in range(len(self.weights[0])):
                cur_neuron = self.weights[row_idx][col_idx]
                distance = np.linalg.norm(winner - cur_neuron)
                if distance <= self.radius:
                    cur_delta = self.lr * distances[row_idx][col_idx]                    
                    self.weights[row_idx][col_idx] += cur_delta
                    delta += abs(cur_delta[0]) + abs(cur_delta[1])

        return delta
    
    def growNode(self, p):
        np.append(self.weights, p)