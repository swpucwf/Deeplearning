import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x,y = iris.data,iris.target

k_range = range(1,31)

k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,x,y,cv=10,scoring="accuracy")
    k_score.append(score.mean())

plt.figure()
plt.plot(k_range,k_score)
plt.xlabel("value of k ")
plt.ylabel("CrossValidation accuracy")
plt.show()