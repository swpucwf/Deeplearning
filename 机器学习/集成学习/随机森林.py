import numpy as np

from sklearn import tree,ensemble,datasets,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

np.random.RandomState(0)

wine = datasets.load_wine()
x,y = wine.data,wine.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

scaler = preprocessing.StandardScaler().fit(x_train)
x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)

clf = ensemble.RandomForestClassifier(n_estimators=25,max_depth=3)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print(accuracy_score(y_test,y_pred))

print(wine.feature_names)
print(clf.feature_importances_)
