"""
Custom simple neural network module
"""
import numpy as np
from scipy.special import expit

class Layer():
    def __init__(self, weights, biases):
        self.weights = np.matrix(weights)
        self.weights = self.weights.astype(float)
        self.biases = np.matrix(biases)
        self.biases = self.biases.astype(float)
        self.num_inputs = self.weights.shape[0]
        self.num_outputs = self.weights.shape[1]

    def threshold(self, applied_biases):
        return expit(applied_biases)

    def run(self, inputs):
        applied_weights = inputs * self.weights
        applied_biases = applied_weights - self.biases
        applied_thresholds = self.threshold(applied_biases)
        return applied_thresholds

    def prnt(self):
        print "weights: "
        print self.weights
        print "biases: "
        print self.biases


class Net():
    def __init__(self, layer_sizes, training_set=None):
        if training_set is None:
            training_set = []
        self.layers = self._make_layers(layer_sizes)
        #TODO validate training_set (modularize training_set functions)
        self.training_set = np.matrix(training_set)

    def _gen_weights(self, prev_layer, cur_layer):
        return [[1] * cur_layer for i in range(prev_layer)]

    def _make_layers(self, layer_sizes):
        layers = []
        for i, layer_size in enumerate(layer_sizes):
            if i > 0:
                prev_layer = layer_sizes[i - 1]
                weights = self._gen_weights(prev_layer, layer_size)
                biases = [[0] * layer_size]
                layer = Layer(weights, biases)
                layers.append(layer)
        return layers

    def set_weight(self, layer_num, output_neuron_num, input_neuron_num, new_weight):
        weight_layer = self.layers[layer_num]
        weight_layer.weights[input_neuron_num, output_neuron_num] = new_weight

    def calc_cost(self):
        example_inputs = self.training_set[0]
        expected_outputs = self.training_set[1]
        actual_outputs = self.run(example_inputs)
        sum_error = actual_outputs - expected_outputs
        sum_squared_error = np.square(sum_error)
        return sum_squared_error

    def run(self, inputs):
        output = inputs
        for i, layer in enumerate(self.layers):
            #print "output from layer " + str(i) + " to " + str(i+1)
            #print output
            output = layer.run(output)
        return output

    def prnt(self):
        for i, layer in enumerate(self.layers):
            print "layer " + str(i) + ":"
            layer.prnt()