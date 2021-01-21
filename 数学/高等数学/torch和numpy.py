import torch
import numpy as np

a = np.array(1)
print(a.shape)

b = torch.tensor(1)
print(b.shape)


a = np.array([1,2,3,4])
print(a.shape)

b = torch.tensor([1,2,3,4])
print(b.shape)


a = torch.tensor([[[[1,2,1],[3,4,1]],[[1,1,2],[1,3,4]]],[[[1,2,1],[3,4,1]],[[1,1,2],[1,3,4]]]])
print(a)
print(a.shape)
a = np.array([[[1,2],[3,4]]])
print(a)
print(a.shape)