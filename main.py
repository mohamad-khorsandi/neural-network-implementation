import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet34
from neural_network import NeuralNetwork


def extract_feature(cifar10):
    resnet = resnet34(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)
    return resnet(cifar10).numpy()


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                 download=True, transform=transform)

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    x_train = []
    y_train = []
    counter = 0
    for img, label in train_data:
        counter += 1
        x_train.append(img)
        y_train.append(label)
        if counter == 10:
            break

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)

    features = extract_feature(x_train)
    train_features = features.reshape(features.shape[0], features.shape[1])
    print(train_features.shape)
    print(train_features)

    neural_network = NeuralNetwork()
    neural_network.train()
    neural_network.confusion_matrix()


if __name__ == '__main__':
    main()