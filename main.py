import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.models import resnet34

from neural_network import NeuralNetwork

def main():
    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                 download=True, transform=transform)
    # You should define x_train and y_train
    feature_extractor = resnet34(pretrained=True)

    #todo
    #neural_network = NeuralNetwork()
    #neural_network.train()
    #neural_network.confusion_matrix()


if __name__ == '__main__':
    main()