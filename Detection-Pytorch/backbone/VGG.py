import torch
from torch import nn
import torch.nn.functional as F
    
class VGG11(nn.Module):
    def __init__(self,num_classes=1000):
        super(VGG11, self).__init__()
        in_dims = 3
        out_dims =64
        layers = []
        for i in range(8):
            layers+=[nn.Conv2d(in_dims,out_dims,3,1,1),nn.ReLU(inplace=True)]
            in_dims = out_dims
            if i in [0,1,3,5,7]:
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
                if i!=5:
                    out_dims*=2
        self.layer = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    def forward(self,x):
        x = self.layer(x)
        x = x.reshape(x.size(0),-1)
        return  self.classifier(x)
class VGG13(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG13, self).__init__()
        in_dims = 3
        out_dims = 64
        layers = []
        for i in range(10):
            layers += [nn.Conv2d(in_dims, out_dims, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dims = out_dims
            if i in [1, 3, 5, 7, 9]:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if i != 7:
                    out_dims *= 2
        self.layer = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):

        x = self.layer(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
class VGG16_1(nn.Module):


    def __init__(self,num_classes=1000):
        super(VGG16_1, self).__init__()
        layers = []
        in_dims = 3
        out_dims = 64

        for i in range(13):
            if i==6:
                layers+=[nn.Conv2d(in_dims,out_dims,1,1,1),nn.ReLU(inplace=True)]
            else:
                layers+=[nn.Conv2d(in_dims,out_dims,3,1,1),nn.ReLU(inplace=True)]
            in_dims = out_dims
            if i in [1,3,6,9,12]:
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
                if i!=9:
                    out_dims*=2

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )
    def forward(self,x):

        x = self.features(x)
        print(x.shape)
        x = x.reshape(x.size(0),-1)
        x = self.classifier(x)
        return x
class VGG16_3(nn.Module):
    def __init__(self,num_classes=1000):
        super(VGG16_3, self).__init__()
        layers = []
        in_dims = 3
        out_dims = 64

        for i in range(13):

            layers+=[nn.Conv2d(in_dims,out_dims,3,1,1),nn.ReLU(inplace=True)]
            in_dims = out_dims
            if i in [1,3,6,9,12]:
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
                if i!=9:
                    out_dims*=2

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )
    def forward(self,x):

        x = self.features(x)
        x = x.reshape(x.size(0),-1)
        x = self.classifier(x)
        return x
class VGG19(nn.Module):

    def __init__(self, num_classes=1000):

        super(VGG19, self).__init__()
        layers = []
        in_dims = 3
        out_dims = 64

        for i in range(16):

            layers += [nn.Conv2d(in_dims, out_dims, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dims = out_dims
            if i in [1, 3, 7, 11, 15]:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if i != 11:
                    out_dims *= 2

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x








if __name__ == '__main__':


    # input = torch.randn(1,3,224,224).cuda()
    # vgg =  VGG16_3().cuda()
    # print(vgg)
    # print(vgg(input).shape)
    # scores = vgg(input)
    # print(scores)
    # input = torch.randn(1, 3, 224, 224).cuda()
    # vgg = VGG11().cuda()
    # print(vgg(input).shape)
    # input = torch.randn(1, 3, 224, 224).cuda()
    # vgg = VGG13().cuda()
    # print(vgg(input).shape)
    # input = torch.randn(1,3,224,224).cuda()
    # vgg =  VGG19().cuda()
    # print(vgg)
    # print(vgg(input).shape)
    net = InceptionV1(3,64,32,64,64,96,32).cuda()
    # print(net)
    input = torch.randn(1,3,256,256).cuda()
    print(net(input).shape)