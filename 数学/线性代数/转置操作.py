import torch

if __name__ == '__main__':
    a = torch.rand(1,4,5,3)
    print(a)
    b = a.permute(0,3,2,1)
    print(b.shape)