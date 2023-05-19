import numpy as np
import torch

import activation_functions


class Dense:
    def __init__(self, input_count, neuron_count, weights, bias):  # todo remove w and b
        self.output = None
        self.b_output = None

        self.input_count = input_count
        self.neuron_count = neuron_count
        # self.weights = np.array(weights)
        # self.bias = np.array(bias)
        self.grad_w = np.ndarray
        self.grad_b = np.ndarray
        self.forward_input = np.ndarray
        self.weights = torch.randn((input_count, neuron_count)).numpy()
        self.bias = torch.randn((1, neuron_count)).numpy()

    def forward(self, inputs):
        activation_functions.assert_data(inputs)
        self.forward_input = inputs
        activation_functions.assert_data(inputs)
        self.output = np.matmul(inputs, self.weights) + self.bias
        activation_functions.assert_data(inputs)


    def backward(self, b_input):
        self.b_output = np.dot(b_input, self.weights.T)
        self.grad_w = -1 * ((np.matmul(np.transpose(self.forward_input), b_input)) / b_input.shape[0])
        bias_derivative = np.full(self.bias.shape, 1)
        self.grad_b = -1 * (np.transpose((bias_derivative * (np.sum(b_input, axis=0))) / b_input.shape[0]))

