import numpy as np

x = np.array([
    [
        [1, 1],
        [1, 1]
    ],
    [
        [2, 2],
        [2, 2]
    ],
    [
        [3, 3],
        [3, 3]
    ],
    [
        [4, 4],
        [4, 4]
    ],
])

if __name__ == '__main__':

    x = x.reshape(2,2,2,2)
    x = x.transpose(2,0,3,1)
    x = x.reshape(4,4)
    print(x)