import torch
from torch import optim
import matplotlib.pyplot as plt
from torch import nn

xs = torch.arange(0,1,0.01)

ys = xs*3 + 4

class Line(nn.Module):

    def __init__(self):
        super().__init__()
        # 模型初始化
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self,x):
        # 前项推导
        return self.w+self.b

if __name__ == '__main__':

    line = Line()
    loss_fn = nn.MSELoss()
    # lr learning rate
    opt = optim.SGD(line.parameters(),lr=0.1)
    plt.ion()
    for epoch in range(1000):
        for _x,_y in zip(xs,ys):
            z = line(_x)
            loss = loss_fn(_y[...,None],z)

            opt.zero_grad()
            loss.backward()
            opt.step()
            print(line.w.item(),line.b.item(),loss.item())
            plt.cla()
            plt.plot(xs,ys,".")
            v = [line.w*e+line.b for e in xs]
            plt.plot(xs,v)
            plt.pause(0.02)
    plt.ioff()
    plt.show()
