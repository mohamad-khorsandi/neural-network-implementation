import copy

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets
from torchvision.models import resnet34

np.random.seed(0)

NEPOCHS = 20
LR = 0.5
DECAY_LR = -0.01
THETA = 1 / 4
DATA_SIZE = None
# r1 = 1
# r2 = 2
FEATURE_SIZE = None
NEURONS = 10
DATA = None


class Neuron:

    def __init__(self, feature_vector):
        self.fv = feature_vector


# i = 0
# while i < DATA_SIZE:
#     x = np.random.uniform(-r2, +r2, FEATURE_SIZE)
#     r = np.linalg.norm(x)
#     if r1 <= r <= r2:
#         DATA[i] = x
#         i += 1
# DATA = torch.tensor(DATA)

# plt.scatter(DATA[:, 0], DATA[:, 1], s=12, label='Data')
# plt.title('Created Dataset')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend()
# plt.show()


def extract_feature(cifar10):
    resnet = resnet34(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)
    return resnet(cifar10).numpy()


def data_cluster(cluster, labels):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6 = []
    cluster7 = []
    cluster8 = []
    cluster9 = []
    cluster10 = []
    cluster_indxs = []
    for idx, c in enumerate(cluster):
        if c == 0:
            cluster1.append(labels[idx])
        elif c == 1:
            cluster2.append(labels[idx])
        elif c == 2:
            cluster3.append(labels[idx])
        elif c == 3:
            cluster4.append(labels[idx])
        elif c == 4:
            cluster5.append(labels[idx])
        elif c == 5:
            cluster6.append(labels[idx])
        elif c == 6:
            cluster7.append(labels[idx])
        elif c == 7:
            cluster8.append(labels[idx])
        elif c == 8:
            cluster9.append(labels[idx])
        elif c == 9:
            cluster10.append(labels[idx])
    print('----------------Number of data from each class in each cluster:----------------')
    print('cluster1:')
    for i in range(10):
        print(f"class {i}: {cluster1.count(i)}")
    print()
    print('cluster2:')
    for i in range(10):
        print(f"class {i}: {cluster2.count(i)}")
    print()
    print('cluster3:')
    for i in range(10):
        print(f"class {i}: {cluster3.count(i)}")
    print()
    print('cluster4:')
    for i in range(10):
        print(f"class {i}: {cluster4.count(i)}")
    print()
    print('cluster5:')
    for i in range(10):
        print(f"class {i}: {cluster5.count(i)}")
    print()
    print('cluster6:')
    for i in range(10):
        print(f"class {i}: {cluster6.count(i)}")
    print()
    print('cluster7:')
    for i in range(10):
        print(f"class {i}: {cluster7.count(i)}")
    print()
    print('cluster8:')
    for i in range(10):
        print(f"class {i}: {cluster8.count(i)}")
    print()
    print('cluster9:')
    for i in range(10):
        print(f"class {i}: {cluster9.count(i)}")
    print()
    print('cluster10:')
    for i in range(10):
        print(f"class {i}: {cluster10.count(i)}")


def t1andt2_neuron_layer(min_bound, max_bound):
    W = np.random.uniform(min_bound, max_bound, (NEURONS, FEATURE_SIZE))
    return W


def type3_neuron_layer(W):
    list_ner = []
    for feature_vec in (W):
        w = Neuron(feature_vec)
        list_ner.append(w)
    W = np.array([[list_ner[0], list_ner[1], list_ner[2]], [list_ner[3], list_ner[4], list_ner[5]],
                  [list_ner[6], list_ner[7], list_ner[8]], [None, list_ner[9], None]])
    return W


def type1(t1_W):
    global LR

    for i in range(NEPOCHS):
        for x in DATA:
            D = np.linalg.norm(t1_W - x, axis=1)
            J = np.argmin(D)
            t1_W[J] += LR * (x - t1_W[J])
            if J == 0:
                t1_W[J + 1] += THETA * LR * (x - t1_W[J + 1])
            elif J == NEURONS - 1:
                t1_W[J - 1] += THETA * LR * (x - t1_W[J - 1])
            else:
                t1_W[J + 1] += THETA * LR * (x - t1_W[J + 1])
                t1_W[J - 1] += THETA * LR * (x - t1_W[J - 1])
        LR += DECAY_LR

    return t1_W


def type2(t2_W):
    global LR
    for i in range(NEPOCHS):
        for x in DATA:
            D = np.linalg.norm(t2_W - x, axis=1)
            J = np.argmin(D)
            t2_W[J] += LR * (x - t2_W[J])
            if J < 3:
                t2_W[J + 1] += THETA * LR * (x - t2_W[J + 1])
                t2_W[J + 2] += THETA * LR * (x - t2_W[J + 2])
                t2_W[J + 3] += THETA * LR * (x - t2_W[J + 3])
            elif J >= NEURONS - 3:
                t2_W[J - 1] += THETA * LR * (x - t2_W[J - 1])
                t2_W[J - 2] += THETA * LR * (x - t2_W[J - 2])
                t2_W[J - 3] += THETA * LR * (x - t2_W[J - 3])
            else:
                t2_W[J + 1] += THETA * LR * (x - t2_W[J + 1])
                t2_W[J + 2] += THETA * LR * (x - t2_W[J + 2])
                t2_W[J + 3] += THETA * LR * (x - t2_W[J + 3])
                t2_W[J - 1] += THETA * LR * (x - t2_W[J - 1])
                t2_W[J - 2] += THETA * LR * (x - t2_W[J - 2])
                t2_W[J - 3] += THETA * LR * (x - t2_W[J - 3])
        LR += DECAY_LR
    return t2_W


def type3(W):
    global LR
    # W = t1andt2_neuron_layer(min_bound, max_bound)
    W_2d = type3_neuron_layer(W)
    for i in range(NEPOCHS):
        for x in DATA:
            D = np.linalg.norm(W - x, axis=1)
            J = np.argmin(D)

            if J == 0:
                W_2d[0][0].fv += LR * (x - W_2d[0][0].fv)
                min_r = 0
                min_c = 0
                W_2d[min_r][min_c + 1].fv += THETA * LR * (x - W_2d[min_r][min_c + 1].fv)
                W_2d[min_r + 1][min_c].fv += THETA * LR * (x - W_2d[min_r + 1][min_c].fv)

            elif J == 1:
                W_2d[0][1].fv += LR * (x - W_2d[0][1].fv)
                min_r = 0
                min_c = 1
                W_2d[min_r][min_c + 1].fv += THETA * LR * (x - W_2d[min_r][min_c + 1].fv)
                W_2d[min_r + 1][min_c].fv += THETA * LR * (x - W_2d[min_r + 1][min_c].fv)
                W_2d[min_r][min_c - 1].fv += THETA * LR * (x - W_2d[min_r][min_c - 1].fv)
            elif J == 2:
                W_2d[0][2].fv += LR * (x - W_2d[0][2].fv)
                min_r = 0
                min_c = 2
                W_2d[min_r + 1][min_c].fv += THETA * LR * (x - W_2d[min_r + 1][min_c].fv)
                W_2d[min_r][min_c - 1].fv += THETA * LR * (x - W_2d[min_r][min_c - 1].fv)
            elif J == 3:
                W_2d[1][0].fv += LR * (x - W_2d[1][0].fv)
                min_r = 1
                min_c = 0
                W_2d[min_r][min_c + 1].fv += THETA * LR * (x - W_2d[min_r][min_c + 1].fv)
                W_2d[min_r + 1][min_c].fv += THETA * LR * (x - W_2d[min_r + 1][min_c].fv)
                W_2d[min_r - 1][min_c].fv += THETA * LR * (x - W_2d[min_r - 1][min_c].fv)
            elif J == 4:
                W_2d[1][1].fv += LR * (x - W_2d[1][1].fv)
                min_r = 1
                min_c = 1
                W_2d[min_r][min_c + 1].fv += THETA * LR * (x - W_2d[min_r][min_c + 1].fv)
                W_2d[min_r][min_c - 1].fv += THETA * LR * (x - W_2d[min_r][min_c - 1].fv)
                W_2d[min_r + 1][min_c].fv += THETA * LR * (x - W_2d[min_r + 1][min_c].fv)
                W_2d[min_r - 1][min_c].fv += THETA * LR * (x - W_2d[min_r - 1][min_c].fv)
            elif J == 5:
                W_2d[1][2].fv += LR * (x - W_2d[1][2].fv)
                min_r = 1
                min_c = 2
                W_2d[min_r + 1][min_c].fv += THETA * LR * (x - W_2d[min_r + 1][min_c].fv)
                W_2d[min_r - 1][min_c].fv += THETA * LR * (x - W_2d[min_r - 1][min_c].fv)
                W_2d[min_r][min_c - 1].fv += THETA * LR * (x - W_2d[min_r][min_c - 1].fv)
            elif J == 6:
                W_2d[2][0].fv += LR * (x - W_2d[2][0].fv)
                min_r = 2
                min_c = 0
                W_2d[min_r][min_c + 1].fv += THETA * LR * (x - W_2d[min_r][min_c + 1].fv)
                W_2d[min_r - 1][min_c].fv += THETA * LR * (x - W_2d[min_r - 1][min_c].fv)
            elif J == 7:
                W_2d[2][1].fv += LR * (x - W_2d[2][1].fv)
                min_r = 2
                min_c = 1
                W_2d[min_r][min_c + 1].fv += THETA * LR * (x - W_2d[min_r][min_c + 1].fv)
                W_2d[min_r][min_c - 1].fv += THETA * LR * (x - W_2d[min_r][min_c - 1].fv)
                W_2d[min_r + 1][min_c].fv += THETA * LR * (x - W_2d[min_r + 1][min_c].fv)
                W_2d[min_r - 1][min_c].fv += THETA * LR * (x - W_2d[min_r - 1][min_c].fv)
            elif J == 8:
                W_2d[2][2].fv += LR * (x - W_2d[2][2].fv)
                min_r = 2
                min_c = 2
                W_2d[min_r - 1][min_c].fv += THETA * LR * (x - W_2d[min_r - 1][min_c].fv)
                W_2d[min_r][min_c - 1].fv += THETA * LR * (x - W_2d[min_r][min_c - 1].fv)
            elif J == 9:
                W_2d[3][1].fv += LR * (x - W_2d[3][1].fv)
                min_r = 3
                min_c = 1
                W_2d[min_r - 1][min_c].fv += THETA * LR * (x - W_2d[min_r - 1][min_c].fv)
        LR += DECAY_LR

    W_random_1 = []
    for i in range(len(W_2d)):
        for j in range(len(W_2d[0])):
            if (i == 3 and j == 0) or (i == 3 and j == 2):
                continue
            W_random_1.append(W_2d[i][j].fv)

    W_random_1 = np.array(W_random_1).reshape(10, 2)

    return W_random_1


def clustering(neuron):
    cluster = np.zeros(DATA.shape[0])
    for i, row in enumerate(DATA):
        D = np.linalg.norm(neuron - row, axis=1)
        idx = np.argmin(D)
        cluster[i] = idx

        # mn_dist = float('inf')
        # # dist of the point from all centroids
        # for idx, centroid in enumerate(neuron):
        #     s = 0
        #     for v in range(features):
        #         s += (float(centroid[v]) - float(row[v])) ** 2
        #     distance = np.sqrt(s)
        #
        #     # store closest centroid
        #     if mn_dist > distance:
        #         mn_dist = distance

    return cluster


def main():
    global DATA_SIZE
    global FEATURE_SIZE
    global DATA

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)

    x_train = []
    y_train = []
    for img, label in train_data:
        x_train.append(img)
        y_train.append(label)

    x_train = torch.stack(x_train)
    y_train = np.array(y_train)

    features = extract_feature(x_train)
    DATA = features.reshape(features.shape[0], features.shape[1])
    DATA_SIZE = features.shape[0]
    FEATURE_SIZE = features.shape[1]
    min_bound = np.min(DATA)
    max_bound = np.max(DATA)

    t1_W = t1andt2_neuron_layer(min_bound, max_bound)
    W1 = copy.deepcopy(t1_W)
    W2 = copy.deepcopy(t1_W)
    W3 = copy.deepcopy(t1_W)

    print('type1:')
    W1_aft = type1(W1)
    print('weigth:')
    print(W1)
    flattened_Weight = W1_aft.reshape(-1, W1_aft.shape[-1])
    print('flatt:')
    print(flattened_Weight)
    tsne = TSNE(n_components=2, perplexity=5)
    tsne_w = tsne.fit_transform(flattened_Weight)
    plt.scatter(tsne_w[:, 0], tsne_w[:, 1])
    plt.show()

    # W2_aft = type2(W2)
    # W3_aft = type3(W3)

    cluster = clustering(W1_aft)
    data_cluster(cluster, y_train)


if __name__ == '__main__':
    main()
