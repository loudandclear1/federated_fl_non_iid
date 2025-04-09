import os

import torch
from torch.utils.data import DataLoader

from model import CNNMnist
from utils import get_mnist

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

def main():
    print("device:", device)
    model_dir = "model"
    _, testset = get_mnist()
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    results = []

    for file in sorted(os.listdir(model_dir)):
        if file.endswith(".pt"):
            parts = file.replace(".pt", "").split("_")
            alpha_val = parts[1].replace("alpha", "")
            mu_val = parts[2].replace("mu", "")
            model_path = os.path.join(model_dir, file)
            model = CNNMnist()
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

            acc = evaluate_model(model, test_loader)
            results.append((alpha_val, mu_val, acc))
            print(f"模型: {file} | α = {alpha_val} | μ = {mu_val} | 准确率: {acc:.2f}%")

    print("\n=== 评估完成 ===")
    print("α 与 μ 的准确率：")
    for alpha, mu, acc in results:
        print(f"α = {alpha.ljust(4)} | μ = {mu.ljust(5)} -> Accuracy = {acc:.2f}%")

if __name__ == "__main__":
    main()