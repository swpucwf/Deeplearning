import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np


x = np.arange(0.0,1.0,0.01)
y = np.sin(2*np.pi*x)
y_ = np.cos(2*np.pi*x)

# 设置宽度为2

plt.rcParams['lines.linewidth'] = 2


plt.plot(x, y,color="r")
plt.show()
# 重置方法
# mp.rcdefaults()

plt.plot(x,y)
plt.show()
mp.rc("lines",linewidth=3)

plt.plot(x,y_,color="g")
plt.show()
