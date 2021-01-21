import torch
import numpy as np

if __name__ == '__main__':

    a = torch.tensor(1)
    print("torch type",type(a))
    b = np.array([1])
    print("numpy type",type(b))

    # a type of  torch  data  convert to a numpy type
    c = torch.from_numpy(b)
    print("type b convert to c",type(c))

    d = np.array(a)
    print("type a convert to d",type(d))

