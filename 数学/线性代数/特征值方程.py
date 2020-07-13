import numpy as np
import torch


if __name__ == '__main__':
    a = torch.tensor([[1,2],[3,4]],dtype=torch.float32)
    print(torch.eig(a,eigenvectors=True))
