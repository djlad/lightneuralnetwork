import numpy as np
from scipy.special import expit

class Layer():
    def __init__(self, weights, biases):
        self.weights = np.array(weights)
        self.weights = self.weights.astype(float)
        self.biases = np.array(biases)
        self.biases = self.biases.astype(float)
        self.num_inputs = self.weights.shape[0]
        self.num_outputs = self.weights.shape[1]

    def threshold(self, applied_weights_biases):
        return expit(applied_weights_biases)

    def threshold_prime(self, applied_weights_biases):
        return np.multiply(self.threshold(applied_weights_biases), (1 - self.threshold(applied_weights_biases)))

    def calc_z(self, inputs):
        applied_weights = np.dot(inputs, self.weights)
        applied_biases = applied_weights - self.biases
        return applied_biases

    def get_z_and_activation(self, inputs):
        #not directly tested
        z = self.calc_z(inputs)
        applied_thresholds = self.threshold(z)
        return z, applied_thresholds

    def run(self, inputs):
        z = self.calc_z(inputs)
        applied_thresholds = self.threshold(z)
        return applied_thresholds

    def prnt(self):
        print "weights: "
        print self.weights
        print "biases: "
        print self.biases