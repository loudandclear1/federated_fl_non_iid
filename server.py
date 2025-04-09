import copy

import torch

from client import train
from model import CNNMnist
from utils import get_mnist, non_iid_partition

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def federated_learning(num_clients=10, num_rounds=20, epochs=5, lr=0.01, mu=0.0, alpha=0.5):
    print("device:", device)
    trainset, testset = get_mnist()
    partition = non_iid_partition(trainset, num_clients=num_clients, alpha=alpha)

    global_model = CNNMnist()
    global_weights = global_model.state_dict()

    for rnd in range(num_rounds):
        local_weights = []
        for i in range(num_clients):
            local_model = copy.deepcopy(global_model)
            w = train(
                local_model,
                trainset,
                partition[i],
                epochs=epochs,
                lr=lr,
                device=device,  # 或你期望使用的设备
                mu=mu,
                global_model=global_model if mu > 0 else None
            )
            local_weights.append(w)

        # FedAvg
        new_state = copy.deepcopy(global_weights)
        for key in new_state:
            new_state[key] = sum(w[key] for w in local_weights) / num_clients
        global_model.load_state_dict(new_state)
        global_weights = new_state

        print(f"Round {rnd + 1} finished.")
    return global_model
