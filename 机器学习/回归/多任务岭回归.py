from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error,mean_squared_error

if __name__ == '__main__':
    x,y =datasets.make_regression(n_samples=100,n_features=1,n_targets=2,noise=10,random_state=0)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
    # 构建模型
    reg = linear_model.MultiTaskLasso(0.1)
    reg.fit(x_train,y_train)
    print(reg.coef_,reg.intercept_)

    y_pred = reg.predict(x_test)

    print(mean_squared_error(y_test,y_pred))
    print(mean_absolute_error(y_test,y_pred))
    print(r2_score(y_test,y_pred))
    print(explained_variance_score(y_test,y_pred))
    _x = np.array([-2.5, 2.5])
    _y = reg.predict(_x[:, None])
    plt.scatter(x_test,y_test[:,0],color="green")
    plt.plot(_x,_y[:,0],linewidth=3,color="red")
    plt.scatter(x_test, y_test[:,1], color="blue")
    plt.plot(_x, _y[:, 1], linewidth=3, color="orange")
    plt.show()