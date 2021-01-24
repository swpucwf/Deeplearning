import torch

a = torch.randn(1, 3, 28, 28)

# 维度，降序
b, index = torch.sort(a, dim=1, descending=True)

print(index.shape)
