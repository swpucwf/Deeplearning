import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.arange(-10,-0.1,0.1)
    y = x**(-1)
    
    plt.plot(x,y)
    plt.show()