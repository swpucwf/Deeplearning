import torch
from torch import nn,optim
import matplotlib.pyplot as plt


class Line(nn.Module):

    def __init__(self):
        super(Line, self).__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))


    def forward(self,x):

        return self.w*x+self.b


if __name__ == '__main__':

    line = Line()
    my_optim = optim.SGD(line.parameters(),lr=0.1)
    _x = torch.arange(0,1,0.01)
    # print(x)
    _y = 3*_x+5+torch.randn(_x.shape)
    plt.ion()
    for _ in range(100):
        for x,y in zip(_x,_y):
            y_pre = line(x)
            loss = (y_pre-y)**2

            my_optim.zero_grad()
            loss.backward()
            my_optim.step()
            print(line.w.item(), line.b.item(), loss.item())

    print(line.w,line.b)









