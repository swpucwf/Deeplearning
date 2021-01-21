import  torch

if __name__ == '__main__':

    x = torch.tensor([3],requires_grad=True,dtype=torch.float32)

    y =  x**3+2

    # 方法一
    # y.backward()
    # print(x.grad)
    # 方法2
    print(torch.autograd.grad(y,x))
