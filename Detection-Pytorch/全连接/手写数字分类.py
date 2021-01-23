import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 权重初始化
def weight_init(network):
    if isinstance(network, nn.Conv2d):
        nn.init.kaiming_normal_(network.weight)
        if network.bias is not None:
            nn.init.zeros_(network.bias)
    if isinstance(network, nn.Linear):
        nn.init.kaiming_normal_(network.weight)
        if network.bias is not None:
            nn.init.zeros_(network.bias)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(0.5,inplace=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(0.5, inplace=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(0.5, inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.outplayer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            # nn.Softmax(-1)
        )
        # 权重初始化
        self.apply(weight_init)

    def forward(self, x):
        conv_result = self.layer(x)
        # print(conv_result.shape)
        n, c, w, h = conv_result.shape
        result = conv_result.reshape(n, c * w * h)
        out_put_result = self.outplayer(result)
        return out_put_result

class Train(object):

    def __init__(self, save_path="./checkpoint"):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.train_Data = datasets.MNIST("./data", train=True, transform=self.tf, download=True)
        self.test_Data = datasets.MNIST("./data", train=False, transform=self.tf, download=True)
        self.train_Data_loader = DataLoader(self.train_Data, batch_size=100, shuffle=True, num_workers=4,
                                            drop_last=True)
        self.test_Data_loader = DataLoader(self.test_Data, batch_size=100, shuffle=True, num_workers=4,
                                           drop_last=True)

        self.model = Model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


        self.opt = optim.Adam([
            {"params": self.model.layer.parameters(), "lr": 0.01},
            {"params": self.model.outplayer.parameters(), "lr": 0.001}, ],
            lr=0.01,
        )
        if os.listdir(self.save_path):
            # 加载存好的路径
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, os.listdir(self.save_path)[-1])))
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, epochs=1000):



        for epoch in range(epochs):
            # 开启训练
            self.model.train()
            sum_loss = 0.
            for i,(img,tag) in enumerate(self.train_Data_loader):
                img,tag = img.to(self.device),tag.to(self.device)

                tag_pred = self.model(img)
                train_loss = self.loss_fn(tag_pred,tag)

                sum_loss+= train_loss.detach().cpu().item()

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
            avg_train_loss = sum_loss/len(self.train_Data_loader)

            # 训练模式
            self.model.eval()

            sum_test_loss = 0.
            score = 0.
            for i, (img, tag) in enumerate(self.test_Data_loader):
                img, tag = img.to(self.device), tag.to(self.device)
                tag_pred = self.model(img)
                test_loss = self.loss_fn(tag_pred, tag)
                tag_det =  torch.argmax(tag_pred,dim=1)

                sum_test_loss+= test_loss.detach().cpu().item()
                score += torch.sum(torch.eq(tag_det,tag)).detach().cpu().item()

            avg_test_loss = sum_test_loss/len(self.test_Data_loader)
            avg_score=score/len(self.test_Data_loader)
            print("epoch",epoch,"avg_train_loss",avg_train_loss,"avg_test_loss",avg_test_loss,"avg_score",avg_score)
            if epoch%5==0:
                torch.save(self.model.state_dict(),"./checkpoint/weight.pth")
                torch.save(self.model,"./checkpoint/model.pth")














if __name__ == '__main__':
    # 用1是不行的，需要在dataloader 上设置drop_Last = True
    # x = torch.randn(20, 1, 28, 28)
    # model = Model()
    # print(model(x).shape)
    train = Train()
    train()