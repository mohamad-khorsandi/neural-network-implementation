import math

import numpy as np


class ReLU:
    def __init__(self):
        self.b_output = None
        self.output = None

    def forward(self, inputs):
        z_array = np.array(inputs)
        row, col = z_array.shape
        for i in range(row):
            for j in range(col):
                z_array[i, j] = max(0, z_array[i, j])
        self.output = z_array

    def backward(self, b_input):
        tmp = (self.output > 0)
        self.b_output = np.multiply(b_input, tmp)

class Sigmoid:
    def forward(self, inputs):
        z_array = np.array(inputs)
        row, col = z_array.shape
        for i in range(row):
            for j in range(col):
                z = z_array[i, j]
                z_array[i, j] = 1.0 / (1.0 + np.exp(-z))
        self.output = z_array
        return self.output

    def backward(self, b_input):
        pass


class Softmax:
    def __init__(self):
        self.b_output = None
        self.output = None

    def forward(self, inputs):
        # # // To do: Implement the softmax formula
        # maxx = np.max(inputs)  # maxx = np.max(inputs, axis=1, keepdims=True)
        # mines = inputs - maxx
        # exp_inputs = np.exp(mines)
        # summ = np.sum(exp_inputs)  # summ = np.sum(exp_inputs, axis=1, keepdims=True)
        # self.output = exp_inputs / summ

        max_inputs = np.max(inputs)
        std_inputs = inputs - max_inputs
        target_exp = np.exp(std_inputs)
        total = np.sum(target_exp, axis=1)
        self.output = np.divide(target_exp, total[:, None])

    def backward(self, b_input):
        self.b_output = b_input

def assert_data(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            assert not math.isnan(data[i][j])
            assert not math.isinf(data[i][j])

if __name__ == '__main__':
    sm = Softmax()
    sm.forward(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
