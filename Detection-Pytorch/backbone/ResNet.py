import torch
from torch import nn

class Bottleneck(nn.Module):

    def __init__(self,in_dims,out_dims,stride=1):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dims,in_dims,kernel_size=1,bias=False),
            nn.BatchNorm2d(in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims,in_dims,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims,out_dims,kernel_size=1,stride=1,bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dims,out_dims,1,1),
            nn.BatchNorm2d(out_dims),
        )

    def forward(self,x):
        out = self.bottleneck(x)
        identity = self.downsample(x)
        out +=identity
        out = self.relu(out)
        return out

if __name__ == '__main__':
    bottleneck = Bottleneck(64,256).cuda()
    print(bottleneck)
    input = torch.randn(1,64,256,256).cuda()
    print(bottleneck(input).shape)