import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = np.arange(-3,3,0.1)
    y = 3**x
    y1 = 0.5**x

    plt.plot(x,y)
    plt.plot(x,y1)
    plt.show()