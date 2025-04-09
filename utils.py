from collections import defaultdict

import numpy as np
from torchvision import datasets, transforms


def get_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    testset = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    return trainset, testset


def non_iid_partition(dataset, num_clients=10, alpha=0.1):
    """使用 Dirichlet 分布进行非IID划分"""
    data_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        data_by_class[label].append(idx)

    class_size = {k: len(v) for k, v in data_by_class.items()}
    client_data_idx = [[] for _ in range(num_clients)]

    for c in range(10):  # 10 classes
        indices = data_by_class[c]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, proportions)
        for i, split in enumerate(splits):
            client_data_idx[i].extend(split.tolist())

    return client_data_idx