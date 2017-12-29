"""
Custom simple neural network module
"""
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


class Net():
    def __init__(self, layer_sizes, training_set=None):
        if training_set is None:
            training_set = [[],[]]
        self.layers = self._make_layers(layer_sizes)
        #TODO validate training_set (modularize training_set functions)
        self.training_set = [np.array(training_set[0]), np.array(training_set[1])]

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
        weight_layer.weights[input_neuron_num][output_neuron_num] = new_weight

    def calc_cost(self):
        example_inputs = self.training_set[0]
        expected_outputs = self.training_set[1]
        actual_outputs = self.run(example_inputs)
        sum_error = actual_outputs - expected_outputs
        sum_squared_error = np.square(sum_error)
        return sum_squared_error

    def get_z_and_activation(self, inputs):
        #not directly tested
        activation = inputs
        zs = []
        activations = []
        for layer in self.layers:
            z, activation = layer.get_z_and_activation(activation)
            zs.append(z)
            activations.append(activation)
        return zs, activations

    def train(self, training_set=None):
        if training_set is None:
            training_set = self.training_set#validate training set here
        example_inputs = training_set[0]
        example_outputs = training_set[1]
        #self.prnt()
        for x, y_expected in zip(example_inputs, example_outputs):
            print 'seperate example'
            zs, activations = self.get_z_and_activation(x)
            gradient = self.calc_gradient(x, y_expected, zs, activations)
            print "gradient"
            print gradient

    def calc_gradient(self, x, y_expected, zs, activations):
        #change in C with respect to neuron inputs * weights + biases(aka Z)
        #this is somewhat unclear, but it implements back propagation to calc gradient
        dCdZ = []
        sp = self.layers[-1].threshold_prime(zs[-1])
        last_layer_dCdZ = np.multiply(sp, activations[-1] - y_expected)
        dCdZ.append(last_layer_dCdZ)
        for layer_num in range(len(zs)-2, -1, -1):
            sp = self.layers[layer_num].threshold_prime(zs[layer_num])
            layer_dCdZ = np.multiply(sp, dCdZ[-1] * self.layers[layer_num+1].weights.transpose())
            dCdZ.append(layer_dCdZ)
        bias_gradient = dCdZ[::-1]
        weight_gradient = []
        print x
        print "i m iter"
        print x[0] *2 
        print "end"
        print self.layers[0].weights
        return bias_gradient

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