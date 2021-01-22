import torch

a = torch.randn(1,3,28,28)

# 改变形状
# print(a.view(28,28,1,3))
# print(a.reshape(28,28,1,3))
# print(a.resize(28,28,1,3))
x = torch.randn(2,3)
x = torch.transpose(x,0,1)
print(x.shape)


x = torch.randn(1,3,28,28)
x = x.permute(2,3,1,0)
print(x.shape)

x = torch.randn(1,28,28)

# 升维度
x = torch.unsqueeze(x,dim=0)
print(x.shape)


# 降维
x = torch.randn(1,28,28)
x = torch.squeeze(x,dim=0)
print(x.shape)
# 扩增维度,添加size = 1

a = torch.randn(1,3,28,28)
print(a.shape)

b = a.expand(6,3,28,28)
c = a.expand_as(b)
print(c.shape)