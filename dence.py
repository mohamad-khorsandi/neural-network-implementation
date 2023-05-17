import numpy as np


class Dense:
    def __init__(self, input_count, neuron_count, weights, bias):  # todo remove w and b
        self.output = None
        self.b_output = None

        self.input_count = input_count
        self.neuron_count = neuron_count
        self.weights = np.array(weights)
        self.bias = np.array(bias)
        self.grad_w = np.ndarray
        self.grad_b = np.ndarray
        self.forward_input = np.ndarray
        # self.weight = torch.randn((n_inputs, n_neurons)).numpy()
        # self.bias = torch.randn((1, n_neurons)).numpy()

    def forward(self, inputs):
        self.forward_input = inputs
        self.output = np.matmul(inputs, self.weights.T) + self.bias

    def backward(self, b_input):
        self.b_output = np.dot(b_input, self.weights)
        self.grad_w = np.matmul(np.traspose(self.forward_input), b_input)
        self.grad_b = b_input
