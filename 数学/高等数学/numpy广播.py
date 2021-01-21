import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.arange(-10,10,0.01)
    y = np.tile(np.array([3]),x.shape)
    print(y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x,y)
    plt.show()