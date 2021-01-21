import torch
from torch import optim
import matplotlib.pyplot as plt

xs = torch.arange(-1,1,0.01)
ys = xs*4+torch.rand(200)/10
# plt.plot(xs,ys,".")
# plt.show()
class Line(torch.nn.Module):

    def __init__(self):
        super(Line, self).__init__()
        # 模型初始化
        self.w = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self,x):
        return self.w*x+self.b


if __name__ == '__main__':
    #     初始化模型
    line = Line()
    # 定义优化器
    opt = optim.SGD(line.parameters(),lr=0.1,momentum=0.1)
    #     训练
    plt.ion()
    for epoch in range(60):
        for _x,_y in zip(xs,ys):
            # 将模型输入到模型中，得到输出
            z = line(_x)
            # 定义损失
            loss =(z-_y)**2

            opt.zero_grad()
            loss.backward()
            opt.step()
            print(line.w.item(),line.b.item(),loss.item())
            plt.cla()
            plt.plot(xs,ys,".")
            v = [line.w*e +line.b for  e in xs]
            plt.plot(xs,v)
            plt.pause(0.01)
        plt.ioff()
        plt.show()

