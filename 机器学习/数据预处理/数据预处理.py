from sklearn import preprocessing
import numpy as np
from sklearn import impute
import  warnings
warnings.filterwarnings("ignore")

x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -3.]])


# 标准化
x_scale = preprocessing.scale(x)
print(x_scale)
print(x_scale.mean(0),x_scale.std(0))

# 标准化
scaler = preprocessing.StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale)
print(x_scale.mean(axis=0),x_scale.std(0))

# minmax
scaler = preprocessing.MinMaxScaler()
x_scale = scaler.fit_transform(x)
print(x_scale)
print(x_scale.mean(0),x_scale.std(0))

# RobustScaler

scaler = preprocessing.RobustScaler()
x_scale = scaler.fit_transform(x)
print(x_scale)
print(x_scale.mean(0),x_scale.std(0))

# Normalizer

scaler = preprocessing.Normalizer(norm="l2")
x_scale = scaler.fit_transform(x)
print(x_scale)
print(x_scale.mean(0),x_scale.std(0))

# 二值化

scaler = preprocessing.Binarizer(threshold=0)
x_scale = scaler.fit_transform(x)
print(x_scale)

# one_hot
enc = preprocessing.OneHotEncoder(sparse=False)
ans= enc.fit_transform([[0],[1],[2],[1]])
print(ans)

imp = impute.SimpleImputer(strategy='mean')
# y_imp = imp.fit_transform([[np.nan, 2], [6, np.nan], [7, 6]])
# print(y_imp)
#
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
y_imp = imp.transform([[np.nan, 2], [6, np.nan], [7, 6]])
print(y_imp)
