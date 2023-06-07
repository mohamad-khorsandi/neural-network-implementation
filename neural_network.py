import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sb
from dence import Dense
from activation_functions import ReLU, Softmax
from loss_functions import CategoricalCrossEntropyLoss
from optimizers import SGD


class NeuralNetwork:
    def __init__(self, first_layer_neurons, hidden_layer_size, last_layer_neurons, x_train, y_train_one_hot, y_train):
        self.Layer1 = Dense(first_layer_neurons, hidden_layer_size)
        self.Act1 = ReLU()
        self.Layer2 = Dense(hidden_layer_size, last_layer_neurons)

        self.Act2 = Softmax()
        self.Loss = CategoricalCrossEntropyLoss()
        self.Optimizer = SGD(learning_rate=2)
        self.x_train = x_train
        self.y_train_one_hot = y_train_one_hot
        self.last_layer_neurons = last_layer_neurons
        self.y_train = y_train

    def train(self):

        for epoch in range(20):
            self.Layer1.forward(self.x_train)
            self.Act1.forward(self.Layer1.output)
            self.Layer2.forward(self.Act1.output)
            self.Act2.forward(self.Layer2.output)
            loss = self.Loss.forward(self.Act2.output, self.y_train_one_hot)

            # Report
            y_predict = np.argmax(self.Act2.output, axis=1)

            print(self.test(self.y_train_one_hot, y_predict))
            accuracy = np.mean(self.y_train == y_predict)
            self.confusion_matrix(self.y_train, y_predict)
            F1_Score = f1_score(self.y_train, y_predict, average='weighted')
            print(f'Epoch:{epoch + 1}')
            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy * 100} %')
            print(f'f1 score: {F1_Score}')
            print('--------------------------')

            # backward
            self.Loss.backward(self.Act2.output, self.y_train_one_hot)
            self.Act2.backward(self.Loss.b_output)
            self.Layer2.backward(self.Act2.b_output)
            self.Act1.backward(self.Layer2.b_output)
            self.Layer1.backward(self.Act1.b_output)

            # update params
            self.Optimizer.update(self.Layer1)
            self.Optimizer.update(self.Layer2)


    def confusion_matrix(self, y_true_one_hot, y_predict):
        cm_train = confusion_matrix(y_true_one_hot, y_predict)
        plt.subplots(figsize=(10, 6))
        sb.heatmap(cm_train, annot=True, fmt='g')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix for the training set")
        plt.show()


    def test(self, y_true, y_pred):
        counter = 0
        for i in range(len(y_true)):
            if y_true[i][y_pred[i]] == 1:
                counter += 1
        return counter
