import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.models import resnet34
from neural_network import NeuralNetwork

def extract_feature():
    resnet = resnet34(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    features = []
    for img, label in cifar10:
        feature = resnet(img.unsqueeze(0))
        features.append(feature.squeeze().detach().numpy())

    features = np.array(features)
    print(features)

def main():
    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    extract_feature()

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                 download=True, transform=transform)
    # You should define x_train and y_train
    x_train = []
    y_train = []

    for img, label in train_data:
        x_train.append(img)
        y_train.append(label)

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)

    print(x_train)
    print(y_train)

    feature_extractor = resnet34(pretrained=True)

    # todo
    #neural_network = NeuralNetwork()
    #neural_network.train()
    #neural_network.confusion_matrix()


if __name__ == '__main__':
    main()