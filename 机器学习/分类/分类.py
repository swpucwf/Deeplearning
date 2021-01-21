from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import linear_model, svm, neighbors, datasets, preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
import warnings

warnings.filterwarnings("ignore")
np.random.RandomState(0)
# 加载数据集
iris = datasets.load_iris()
x,y = iris.data,iris.target

# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
# 数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 创建模型
# clf = linear_model.LogisticRegression()
# clf = linear_model.SGDClassifier()
# clf = linear_model.RidgeClassifier(0.5)
# clf = linear_model.PassiveAggressiveClassifier()
clf = tree.DecisionTreeClassifier(max_depth=1)
# 模型拟合
clf.fit(x_train,y_train)

# 预测
y_pred = clf.predict(x_test)


print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred,average="micro"))
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))