"coding:utf-8"
import torch


a = torch.Tensor([[1,2,3],[4,5,6]])
print(a.shape)

b = torch.chunk(a,3,dim=1)
print(b)

c = torch.split(a,[1,2],dim=1)
print(c)