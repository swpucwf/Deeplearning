import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
rng = np.random.RandomState(0)

X = 5*rng.rand(100,1)
y = np.sin(X).ravel()

y[::5]+=3*(0.5-rng.rand(X.shape[0]//5))

svr = GridSearchCV(SVR(kernel="rbf"),
                   param_grid={"C":[1e0,1e1,1e2,1e3],
                               "gamma":np.logspace(-2,2,5)})

svr.fit(X,y)
print(svr.best_score_,svr.best_params_)
X_plot = np.linspace(0,5,100)
y_svr = svr.predict(X_plot[:,None])
plt.scatter(X,y)
plt.plot(X_plot,y_svr,color="red")
plt.show()