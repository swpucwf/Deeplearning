import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x,y,coef = make_regression(n_samples=100,
                           n_features=1,
                           n_targets=1,
                           noise=3,
                           coef=True)
print(coef)

plt.scatter(x,y)
plt.plot(x,x*coef,color="blue",linewidth=3)
plt.show()