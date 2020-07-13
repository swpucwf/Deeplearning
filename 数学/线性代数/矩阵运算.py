import torch

a = torch.tensor([[1,2],[3,4],[5,6]])
b = torch.tensor([[5,6],[7,8],[9,10]])
c = torch.tensor([[1,2],[3,4]])
d = torch.tensor([1,2,3])
e = torch.tensor([3,4,5])


print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
print(e.shape)

if __name__ == '__main__':

    # 矩阵运算需要同形 , 注意
    print(a+b)
    print(a-b)
    print(a*b)
    print(a+3)
    print(b+3)
    print(a*3)
    # a,c
    print(a)
    print(c)
    print(a@c)
    print(torch.matmul(a,c))
    print(a)
    print(a.T)
    print(a.t())
    x = torch.arange(0,24,1).reshape(-1,2,3)
    print(x)
    x = x.permute(2,1,0)
    print(x)