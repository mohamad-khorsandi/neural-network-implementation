import numpy as np
from load_and_extract_utils import get_train_features
from neural_network import NeuralNetwork

def main():
    x_train, y_train = get_train_features()
    y_train_one_hot = one_hot_encode(y_train, 10)

    x_train = normalize_data(x_train)
    print('normalized')
    neural_network = NeuralNetwork(x_train.shape[1], 20, 10, x_train, y_train_one_hot, y_train)
    neural_network.train()
    # neural_network.confusion_matrix()


def normalize_data(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    # Normalize the data
    data_norm = (data - data_min) / (data_max - data_min)

    return data_norm


def one_hot_encode(data_class, cat_count):
    one_hot_list = np.zeros((len(data_class), cat_count))

    for i in range(len(data_class)):
        one_hot = np.zeros(cat_count)
        one_hot[data_class[i]] = 1
        one_hot_list[i] = one_hot

    return one_hot_list  # TODO


if __name__ == '__main__':
    main()
    # train_features = np.array([[1, 2, 3, 1],
    #                            [4, 5, 6, 8]])
    # y_train = one_hot_encode([0, 1], 2)
    #
    # neural_network = NeuralNetwork(4, 3, 2, train_features, y_train)
    # neural_network.train()

