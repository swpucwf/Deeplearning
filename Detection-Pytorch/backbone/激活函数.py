import torch

def Sigmoid(x):
    y = 1/(1+torch.exp(x))
    return y


def tanh(x):
    y = (torch.exp(x)+torch.exp(-x))/(torch.exp(x)-torch.exp(-x))

    return y

def Relu(x):

    return x if x>=0 else 0

def leakyRelu(x,a=0.1):
    return x if x>=0 else (1/a)*x


def Softmax(x):

    return torch.exp(x)/torch.sum(torch.exp(x),dim=1)

x = torch.randn(1,4)
print(Softmax(x))
