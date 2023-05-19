import numpy as np
import torch

import activation_functions
from assets import assert_mat, assert_vac


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
        self.bias = torch.randn((1, neuron_count)).numpy() # todo chekc bias

    def forward(self, inputs):
        self.forward_input = inputs
        self.output = np.matmul(inputs, self.weights) + self.bias

    def backward(self, b_input):
        assert b_input.shape[0] != 0
        self.b_output = np.dot(b_input, self.weights.T)
        self.grad_w = -1 * ((np.matmul(np.transpose(self.forward_input), b_input)) / b_input.shape[0])
        bias_derivative = np.full(self.bias.shape, 1)
        self.grad_b = -1 * (np.transpose((bias_derivative * (np.sum(b_input, axis=0))) / b_input.shape[0]))

