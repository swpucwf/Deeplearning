import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # generate data of x
    x = np.arange(-10,10,0.01)
    y = x**2
    # 连续
    plt.plot(x,y,color="red")
    # 显示
    plt.show()