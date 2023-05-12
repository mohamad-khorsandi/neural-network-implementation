from cmath import log

import numpy as np


class CategoricalCrossEntropyLoss:
    def __init__(self):
        self.b_output = None

    def forward(self, softmax_output, y_one_hot):
        for i in range(len(softmax_output)):
            single_loss = - np.sum(y_one_hot[i] * np.log(softmax_output[i]))
            print(y_one_hot[i])
            print(softmax_output[i])
            print(y_one_hot[i] * np.log(softmax_output[i]))
            print("--------------------------------")
        return np.sum(single_loss) / softmax_output.shape[0]


    def backward(self, softmax_output, class_label):
        class_label = self.to_1hot(class_label)
        pass
        # // To do: Implement the CCE loss derivative with respect to predicted label
