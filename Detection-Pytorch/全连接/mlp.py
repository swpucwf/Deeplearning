from torch import nn,optim
import torch
class MLP(nn.Module):

    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim1),
            nn.ReLU(),
            nn.Linear(hid_dim1,hid_dim2),
            nn.ReLU(),
            nn.Linear(hid_dim2,out_dim),
            nn.ReLU()
        )
        self.out_put = nn.Softmax(dim=-1)

    def forward(self,x):
        rx = self.layer(x)
        return self.out_put(rx)


if __name__ == '__main__':

    mlp = MLP(784,40,40,10)
    x = torch.randn(1,784)
    y = torch.zeros(10)
    opt = optim.Adam([{"params":mlp.layer.parameters(),"lr":0.001},
                      { "params":mlp.out_put.parameters(),"lr":0.000001},
                      ],lr=0.01)
    y_pred = mlp(x)
    # loss 一般是个标量哈
    loss = torch.mean((y-y_pred)**2)
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()



