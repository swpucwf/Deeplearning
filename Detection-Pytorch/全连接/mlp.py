from torch import nn
import numpy as np
import torch
# 构造数据

x = torch.arange(0.01,1,100)
print(x)
y = [3*i+4+np.random.rand(99)/100 for i in x]

print(y)

class MLP(nn.Module):


    def __init__(self):

        super(MLP, self).__init__()
        self.layer = nn.Sequential(
        )

    def forward(self,x):

        pass