import os

import torch

from server import federated_learning

if __name__ == "__main__":
    alpha_list = [0.1, 0.3, 0.5, 0.7, 1.0]  # 控制异质性程度
    mu_list = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]  # 控制正则项强度

    for alpha in alpha_list:
        for mu in mu_list:
            filename = f"model/fedprox_alpha{alpha}_mu{mu:.3f}_model.pt"
            if os.path.exists(filename):
                print(f"模型已存在，跳过训练：{filename}")
                continue
            print(f"开始训练：α = {alpha}, μ = {mu:.3f}")
            model = federated_learning(mu=mu, alpha=alpha)
            torch.save(model.state_dict(), filename)
            print(f"模型已保存为 {filename}")
