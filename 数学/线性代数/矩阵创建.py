import numpy as np
import torch


if __name__ == '__main__':
    # 对角阵
    a  = np.diag([1,2,3,4])
    print(a)
    b = torch.diag(torch.tensor([1,2,3,4]))
    print(b)
    c = np.eye(3,4)
    print(c)
    d = torch.eye(3,4)
    print(d)
    e = np.tri(3,3)
    print(e)
    # 下三角
    f = torch.tril(torch.ones(3,3))
    print(f)

    g = np.zeros((3,3))
    print(g)

    h = torch.zeros(3,3)
    print(h)


