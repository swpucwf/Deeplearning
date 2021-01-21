import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import neighbors,datasets,preprocessing

# 加载数据
iris = datasets.load_iris()
x,y = iris.data,iris.target
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
# 数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 创建模型
knn = neighbors.KNeighborsClassifier(n_neighbors=12)
# 模型拟合
knn.fit(x_train,y_train)
# 交叉验证
scores = cross_val_score(knn,x_train,y_train,cv=10,scoring="accuracy")
print(scores)
print(scores.mean())

y_pred = knn.predict(x_test)
print(accuracy_score(y_pred=y_pred,y_true=y_test))