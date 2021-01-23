from torch import nn
import numpy as np
import torch
# 构造数据
from torch import optim

x = torch.arange(0,1,0.01)
y = x*3+4+np.random.rand(100)/100

class Linear(nn.Module):

    def __init__(self):

        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.rand(1),requires_grad=True)
        self.b = nn.Parameter(torch.rand(1),requires_grad=True)

    def forward(self,x):

        return self.w*x+self.b

if __name__ == '__main__':
    line = Linear()
    opt  = optim.Adam(line.parameters())

    for i in range(1000):

        for _x,_y in zip(x,y):

            y_pred = line(_x)
            loss = (y_pred-_y)**2

            opt.zero_grad()
            loss.backward()
            opt.step()
            print(line.w, line.b)


