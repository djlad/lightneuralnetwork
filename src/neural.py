"""
Custom simple neural network module
"""
from itertools import izip
import numpy as np
from layer import Layer
from scipy.special import expit

class Net():
    def __init__(self, layer_sizes, training_set=None):
        if training_set is None:
            training_set = [[], []]
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
        #sum_squared_error = np.sqrt(sum_squared_error)
        return np.sum(sum_squared_error)

    def get_z_and_activation(self, inputs):
        #not directly tested
        activation = inputs
        zs = []
        activations = [inputs]
        for layer in self.layers:
            z, activation = layer.get_z_and_activation(activation)
            zs.append(z[0])
            activations.append(activation[0])
        return zs, activations

    def randomize_wb(self):
        for i, layer in enumerate(self.layers):
            b_shape = layer.biases.shape
            w_shape = layer.weights.shape
            layer.biases = np.random.rand(*b_shape)
            layer.weights = np.random.rand(*w_shape)

    def train(self, training_set=None, learning_rate=.0005):
        self.randomize_wb()
        last_er = float("inf")
        for i in range(50000*2):
            if training_set is None:
                training_set = self.training_set#validate training set here
            example_inputs = training_set[0]
            example_outputs = training_set[1]
            sum_weight_gradient = None
            sum_bias_gradient = None
            for x, y_expected in zip(example_inputs, example_outputs):
                zs, activations = self.get_z_and_activation(x)
                weight_gradient, bias_gradient = self.calc_gradient(x, y_expected)
                if sum_weight_gradient is None:
                    sum_weight_gradient = weight_gradient
                    sum_bias_gradient = bias_gradient
                else:
                    sum_weight_gradient += weight_gradient
                    sum_bias_gradient += bias_gradient

            layers_gradients = izip(self.layers, sum_weight_gradient, sum_bias_gradient)
            for layer, layer_weight_gradient, layer_bias_gradient in layers_gradients:
                layer.weights = layer.weights + layer_weight_gradient * learning_rate
                layer.biases = layer.biases + layer_bias_gradient * learning_rate

            if (i+1) % 2000 == 0:
                print self.calc_cost()
        print "training cycle completed"

    def calc_gradient(self, x, y_expected):
        #change in C with respect to neuron inputs * weights + biases(aka Z)
        #this is unclear, but it implements back propagation to calc gradient
        zs, activations = self.get_z_and_activation(x)
        dCdZ = []
        sp = self.layers[-1].threshold_prime(zs[-1])
        last_layer_dCdZ = np.multiply(sp, activations[-1] - y_expected)
        dCdZ.append(last_layer_dCdZ)
        for layer_num in range(len(zs)-2, -1, -1):
            sp = self.layers[layer_num].threshold_prime(zs[layer_num])
            layer_dCdZ = np.multiply(sp, np.dot(dCdZ[-1], self.layers[layer_num+1].weights.transpose()))
            dCdZ.append(layer_dCdZ)
        bias_gradient = dCdZ[::-1]
        weight_gradient = []
        for i in range(0, len(activations)-1):
            activation = activations[i]
            weight_gradient.append([])
            for neuron_input in activation:
                weight_gradient[-1].append((neuron_input * bias_gradient[i]))
            weight_gradient[-1] = np.array(weight_gradient[-1])
        return np.array(weight_gradient), np.array(bias_gradient)

    def run(self, inputs):
        output = inputs
        for i, layer in enumerate(self.layers):
            output = layer.run(output)
        return output

    def prnt(self):
        for i, layer in enumerate(self.layers):
            print "layer " + str(i) + ":"
            layer.prnt()