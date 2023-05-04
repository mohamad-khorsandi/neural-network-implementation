class CategoricalCrossEntropyLoss:
    def __init__(self):
        self.b_output = None

    def forward(self, softmax_output, class_label):
        class_label = self.to_1hot(class_label)
        # // To do: Implement the CCE loss formula

    def backward(self, softmax_output, class_label):
        class_label = self.to_1hot(class_label)
        pass
        # // To do: Implement the CCE loss derivative with respect to predicted label

    def to_1hot(self, class_label):
        pass
