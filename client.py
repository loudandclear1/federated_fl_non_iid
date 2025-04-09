import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

def train(model, dataset, idxs, epochs=5, lr=0.01, device='cpu', mu=0.0, global_model=None):
    if global_model is not None:
        global_model.to(device)
    model.to(device)
    model.train()
    loader = DataLoader(Subset(dataset, idxs), batch_size=51200, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in loader:
            data = data.to(device).float()
            target = target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            if mu > 0.0 and global_model is not None:
                # FedProx 正则项
                prox_reg = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    prox_reg += ((w - w_t.detach()) ** 2).sum()
                loss += mu / 2 * prox_reg
            loss.backward()
            optimizer.step()
    return model.state_dict()