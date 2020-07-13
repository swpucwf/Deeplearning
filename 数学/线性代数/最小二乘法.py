import numpy as np
import torch

if __name__ == '__main__':
    # x = np.matrix(np.array([[3],[1],[6]],dtype=np.float32))
    # y = 4*x
    # print(x)
    # print(y)
    # print(":=================")
    # print((x.T@x).I@x.T@y)

    a = torch.tensor([[1., 2.], [3., 4.]])
    # b = a.t()
    # print(a@b)
    print(torch.inverse(a))

    b = np.array([[1., 2.], [3., 4.]])
    print(np.linalg.inv(b))

    # c = np.matrix(np.array([[1., 2.], [3., 4.]]))
    # print(c.I)