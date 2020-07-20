import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
x = np.sort(5*np.random.rand(40,1),axis=0)
T = np.linspace(0,5,500)[:,np.newaxis]
y = np.sin(x).ravel()

y[::5]+=1*(0.5-np.random.rand(8))
n_neighbor = 5
for i,weights in enumerate(["uniform","distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbor,weights=weights)
    _y = knn.fit(x,y).predict(T)
    plt.subplot(2,1,i+1)
    plt.scatter(x,y,c="k",label="Data")
    plt.plot(T,_y,c="g",label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title('knn(k=%i,weights = "%s")'%(n_neighbor,weights))
plt.show()