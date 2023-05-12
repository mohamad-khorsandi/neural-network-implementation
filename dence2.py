import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons, weight_arr, bias_arr):
        self.b_output = None
        self.output = None
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.array(weight_arr)  # weight arr = [ [W11 ... W120], ... ,[Wn1 ... Wn20] ]
        self.bias = np.array(bias_arr)

    def forward(self, inputs):  # input = x_train
        output_array = np.zeros(self.n_inputs, self.n_neurons)  # output araay = [ [xiWi1 ... xiWi20], ... ,[xiWi1 ... xiWi20] ]
        x_train = np.array(inputs)
        for i in range(x_train.ndim):
            input_attributes = np.array(x_train[i])
            output_array[i] = np.dot(input_attributes, self.weights) + self.bias
        self.output = output_array
        return self.output

    def backward(self, b_input):
        pass
        # // To do: Weight and bias gradients
