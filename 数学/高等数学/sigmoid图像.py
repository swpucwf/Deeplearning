import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = np.arange(-10,10,0.1)
    y = 1/(1+np.exp(-x))
    y1 = np.zeros_like(x)+0.5
    plt.plot(x,y)
    plt.plot(x,y1)
    plt.show()