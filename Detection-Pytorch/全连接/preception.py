import torch
from torch import nn

class Linear(nn.Module):


    def __init__(self,in_dim,out_dim):

        super(Linear, self).__init__()
        self.w =  nn.Parameter(torch.randn(in_dim,out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self,x):
        x = torch.matmul(x,self.w)
        y  = x + self.b.expand_as(self.x)
        return y

class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer =nn.Sequential(
            Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            Linear(hid_dim,out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)



