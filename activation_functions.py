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
        target_exp = np.exp(inputs)
        self.resolve_inf(target_exp)
        # assert_mat(target_exp)

        total = np.sum(target_exp, axis=1)
        self.resolve_inf(total)
        # assert_vec(total)

        self.output = np.divide(target_exp, total[:, None])

    def backward(self, b_input):
        self.b_output = b_input

    def resolve_inf(self, mat):
        Max = np.finfo(np.float32).max
        if len(mat.shape) == 2:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if math.isinf(mat[i][j]):
                        mat[i][j] = Max

        elif len(mat.shape) == 1:
            for i in range(len(mat)):
                if math.isinf(mat[i]):
                    mat[i] = Max
