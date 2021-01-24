import torch
from torch import nn
import torch.nn.functional as F

# 单独InceptionV2
class BasicConv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels,eps=0.001)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x,inplace=True)

class InceptionV2(nn.Module):

    def __init__(self):
        super(InceptionV2, self).__init__()
        self.branch1 = BasicConv2d(192,96,1,0)
        # 1x1 3x3卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 48, 1, 0),
            BasicConv2d(48,64,(1,3),1),
            BasicConv2d(64,64,(3,1),0)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(192, 64, 1, 0),
            BasicConv2d(64, 96, (1,3), 0),
            BasicConv2d(96, 96, (3,1), 1),
            BasicConv2d(96, 96, (1, 3), 0),
            BasicConv2d(96, 96, (3, 1), 1)
        )
        # 平均池化和1x卷积分支
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3,stride=1,padding=1,count_include_pad=False),
            BasicConv2d(192,64,1,0)
        )

    def forward(self,x):

        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)
        print(x0.shape,x1.shape,x2.shape,x3.shape)
        out = torch.cat([x0,x1,x2,x3],dim=1)
        return out



if __name__ == '__main__':
    model = InceptionV2().cuda()
    print(model)
    # # 默认输入
    input =  torch.randn(1,192,32,32).cuda()
    print(model(input).shape)
