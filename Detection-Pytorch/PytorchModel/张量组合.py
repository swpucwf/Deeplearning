import torch
# 拼接 cat方法
a = torch.Tensor([[1,2],[3,4]])
b = torch.Tensor([[5,6],[7,8]])
print(a.shape)
print(b.shape)
print(a,b)
# cat
# c = torch.cat([a,b],dim=0)
# print(c.shape)
# print(c)
# d = torch.cat([a,b],dim=1)
# print(d.shape)
# print(d)
# f = torch.cat([a,b])
# print(f.shape)
# stack 方法
# 会添加一个维度
c = torch.stack([a,b],dim=0)
print(c.shape)
print(c)

d = torch.stack([a,b],dim=1)
print(d.shape)
print(d)
f = torch.stack([a,b],dim=2)
print(f.shape)
print(f)