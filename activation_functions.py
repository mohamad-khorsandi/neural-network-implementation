import numpy as np


class ReLU:
    def __init__(self):
        self.b_output = None
        self.output = None

    def forward(self, inputs):  # input = arrays of z
        z_array = np.array(inputs)
        row, col = z_array.shape
        for i in range(row):
            for j in range(col):
                z_array[i, j] = max(0, z_array[i, j])
        self.output = z_array
        return self.output

    def backward(self, b_input):
        pass
        # // To do: Implement the ReLU derivative with respect to the input


class Sigmoid:
    def forward(self, inputs):
        pass
        # // To do: Implement the sigmoid formula

    def backward(self, b_input):
        pass
        # // To do: Implement the sigmoid derivative with respect to the input


class Softmax:
    def __init__(self):
        self.b_output = None
        self.output = None

    def forward(self, inputs):
        # // To do: Implement the softmax formula
        cache = np.exp(inputs)
        self.output = cache / np.sum(cache, axis=0)
        return self.output

    def backward(self, b_input):
        # // To do: Implement the softmax derivative with respect to the input
        self.b_output = b_input * (self.output - self.output ** 2)


if __name__ == '__main__':
    sm = Softmax()
    sm.forward(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
