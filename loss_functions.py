import math

import numpy as np


class CategoricalCrossEntropyLoss:
    def __init__(self):
        self.b_output = None

    def forward(self, softmax_output, y_one_hot):
        sum_loss = 0
        epsilon = 1e-10
        softmax_output = np.clip(softmax_output, epsilon, 1. - epsilon)
        for i in range(len(softmax_output)):
            single_loss = np.sum(y_one_hot[i] * np.log(softmax_output[i]))
            sum_loss += single_loss
        assert not softmax_output.shape ==0
        assert not math.isnan(-1 * (sum_loss / softmax_output.shape[0]))
        return -1 * (sum_loss / softmax_output.shape[0])

    def backward(self, softmax_output, class_label):
        self.b_output = class_label - softmax_output
