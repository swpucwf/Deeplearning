import torch

# 随机生成形状的张量
x = torch.Tensor(2, 2).float()

y = torch.DoubleTensor(1, 4)
print(x)
print(y)

c = torch.Tensor([[1, 2], [2, 3]])
print(c)

# 全0 矩阵

_x = torch.zeros(2, 2)
print(_x)

_y = torch.ones(2, 2)
print(_y)

# 对角阵
_c = torch.eye(2, 2)
print(_c)

_z = torch.randn(1, 2, 3)
print(_z)

_f = torch.randperm(10)
print(_f)
