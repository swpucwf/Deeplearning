import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# 希尔伯特矩阵
x = 1./(np.arange(1,11)+np.arange(0,10)[:,np.newaxis])
y = np.ones(10)
# x = (np.arange(1,11)+np.arange(0,10)[:,np.newaxis])
# print(x)
# 计算不同岭系数时的回归系数
n_alphas = 200

alphas = np.logspace(-10,-2,n_alphas)
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a,fit_intercept=False)
    ridge.fit(x,y)
    coefs.append(ridge.coef_)

# 绘图
ax = plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel("岭回归系数alpha")
plt.ylabel("回归系数coef_")
plt.title("岭系数对回归系数的影响")
plt.axis("tight")
plt.show()