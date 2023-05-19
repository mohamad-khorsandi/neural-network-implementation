from dence import Dense


class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, layer: Dense):
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                layer.weights[i, j] = layer.weights[i, j] - (self.learning_rate * layer.grad_w[i, j])
        for j in range(layer.bias.shape[0]):
            layer.bias[j] = layer.bias[j] - (self.learning_rate * layer.grad_b[j])


