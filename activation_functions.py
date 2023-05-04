class ReLU:
    def __init__(self):
        self.b_output = None
        self.output = None

    def forward(self, inputs):
        pass
        # // To do: Implement the ReLU formula

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
        pass
        # // To do: Implement the softmax formula

    def backward(self, b_input):
        pass
        # // To do: Implement the softmax derivative with respect to the input
