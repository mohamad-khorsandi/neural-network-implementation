import math

import numpy as np

from assets import assert_vac, assert_mat
from dence import Dense
from activation_functions import ReLU, Sigmoid, Softmax
from loss_functions import CategoricalCrossEntropyLoss
from optimizers import SGD


class NeuralNetwork:
    def __init__(self, first_layer_neurons, hidden_layer_size, last_layer_neurons, x_train, y_train):
        w1 = [[1, 1, 1, 1],
              [-1, -1, -1, -1],
              [2, 2, 2, 2]]
        b1 = [0, 3, 2]

        self.Layer1 = Dense(first_layer_neurons, hidden_layer_size, None, None)

        self.Act1 = ReLU()

        w2 = [[0, 1, 3],
              [1, 0, 1]]
        b2 = [-1, 2]
        self.Layer2 = Dense(hidden_layer_size, last_layer_neurons, None, None)

        self.Act2 = Softmax()
        self.Loss = CategoricalCrossEntropyLoss()
        self.Optimizer = SGD(learning_rate=0.001)
        self.x_train = x_train
        self.y_train = y_train
        self.last_layer_neurons = last_layer_neurons

    def train(self):
        for epoch in range(20):
            # forward
            assert_mat(self.x_train)

            self.Layer1.forward(self.x_train)
            assert_mat(self.Layer1.output)

            self.Act1.forward(self.Layer1.output)
            assert_mat(self.Act1.output)

            self.Layer2.forward(self.Act1.output)
            assert_mat(self.Layer2.output)

            self.Act2.forward(self.Layer2.output)
            assert_mat(self.Act2.output)

            loss = self.Loss.forward(self.Act2.output, self.y_train)
            assert not math.isnan(loss)
            # Report
            y_predict = np.argmax(self.Act2.output, axis=1)
            accuracy = np.mean(self.y_train == y_predict)
            print(f'Epoch:{epoch}')
            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy}')
            print('--------------------------')

            # backward
            self.Loss.backward(self.Act2.output, self.y_train)
            assert_mat(self.Loss.b_output)

            self.Act2.backward(self.Loss.b_output)
            assert_mat(self.Act2.b_output)

            self.Layer2.backward(self.Act2.b_output)
            assert_mat(self.Layer2.b_output)

            self.Act1.backward(self.Layer2.b_output)
            assert_mat(self.Act1.b_output)

            self.Layer1.backward(self.Act1.b_output)
            assert_mat(self.Layer1.b_output)


            # update params
            self.Optimizer.update(self.Layer1)
            self.Optimizer.update(self.Layer2)

    # todo cal this for test and train set
    def confusion_matrix(self, y_true, y_predict):
        pass
        # cm_train = confusion_matrix(y_true, y_predict)
        # plt.subplots(figsize=(10, 6))
        # sb.heatmap(cm_train, annot=True, fmt='g')  # todo make sure that "sb" is supposed to be seaborn
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Confusion Matrix for the training set")
        # plt.show()
