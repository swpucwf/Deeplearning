import torch

a = torch.tensor([[[1,2,1],[1,3,4]],[[1,1,2],[3,1,4]]])
b = torch.tensor([[[1],[3],[5]],[[2],[4],[7]]])

print(a.shape)
print(b.shape)
print(torch.matmul(a,b).shape)