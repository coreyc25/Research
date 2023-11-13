import unittest
from neuron import Neurons
import numpy as np
import math

class TestNeuron(unittest.TestCase):

    def setUp(self):
        neuron_size = 10
        features = 2
        neighborhood_radius = 50
        learning_rate = 0.02
        neurons = Neurons(neuron_size, features, neighborhood_radius, learning_rate)
        self.neurons = neurons

    # Test to ensure the Neurons object is created correctly
    def test_init(self):
        neurons = self.neurons

        self.assertEqual(neurons.size, 10)
        self.assertEqual(neurons.count, 100)
        self.assertEqual(neurons.radius, 50)
        self.assertEqual(neurons.lr, 0.02)
        self.assertEqual(neurons.iteration, 0)

        weights_row_len = len(neurons.weights)
        weights_col_len = len(neurons.weights[0])
        weights_feature_len = len(neurons.weights[0][0])

        self.assertEqual(weights_row_len, 10)
        self.assertEqual(weights_col_len, 10)
        self.assertEqual(weights_feature_len, 2)

    """
    Test the findWinner function to ensure the following:
        error is correctly calculated and returned
        winner is correctly calculated and returned
    """
    def test_findWinner(self):
        neurons = self.neurons

        p = [10, 10]
        neurons.weights = [[[500, 500], [100, 100], [4, 5]], 
                           [[15, 20], [20, 40], [50, 80]],
                           [[10, 15], [10, 5], [5, 10]]]

        winner, error = neurons.FindWinner(p)

        self.assertEqual(winner, neurons.weights[2])
        self.assertEqual(error, math.sqrt(75))

    """
    Test the adaptWeights function to ensure the following:
        delta is correctly calculated and returned
        neurons.weights is correctly changed
    """
    def test_adaptWeights(self):
        neurons = self.neurons
        neurons.weights = np.array([[[500, 500], [100, 100], [50, 5.0]], 
                           [[15, 20], [20, 40], [50, 80]],
                           [[10, 15], [10, 5], [5, 10]]])
        # print(neurons.weights)

        winner = [[10, 15], [10, 5], [5, 10]]
        p = [10, 10]
        delta = neurons.adaptWeights(winner, p)
        self.assertEqual(delta, 5000)


if __name__ == '__main__':
    unittest.main()