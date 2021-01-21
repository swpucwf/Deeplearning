import numpy as np
from sklearn import ensemble,neighbors,datasets,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

np.random.RandomState(0)
#
# wine = datasets.load_wine()
#
# x,y = wine.data,wine.target
data_s = []
with open("label.txt","r") as f:
    data = f.readline()
    while data:
        data_s.append(np.array([float(i) for i in data.split()]))
        data = f.readline()

data_s = np.array(data_s)
print(data_s.shape)
#
x,y = data_s[:,:-1],data_s[:,-1]
#

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# scaler = preprocessing.MinMaxScaler().fit(x_train)
# #
# x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)

clf = ensemble.BaggingRegressor(neighbors.KNeighborsRegressor(),n_estimators=50,max_features=0.5,max_samples=0.5)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))