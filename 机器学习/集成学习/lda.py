from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis


iris = load_iris()
x,y = iris.data,iris.target

lda3 = LinearDiscriminantAnalysis(n_components=3)
lda3.fit(x,y)
x3 = lda3.transform(x)
print(lda3.explained_variance_ratio_)
plt.scatter(x3[:,0],x3[:,1],s=88,c=y,alpha=0.5)
plt.show()