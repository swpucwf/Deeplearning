import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    x = np.arange(0.1,10,0.1)
    y = np.log(x)
    y1 = np.zeros(x.shape)
    print(y1)
    # 同形的zeros
    y1 = np.zeros_like(x)
    print(y1)
    y2 = np.log(x)/np.log(np.array([0.5]))
    plt.plot(x,y)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()
