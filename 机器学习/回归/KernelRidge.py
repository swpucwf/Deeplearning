# import numpy as np
# from sklearn.kernel_ridge import KernelRidge
# import matplotlib.pyplot as plt
# from  sklearn.model_selection import GridSearchCV
#
# rng = np.random.RandomState(0)
#
# x = 5*rng.rand(100,1)
# y = np.sin(x).ravel()
# # 加噪音
# y[::5]+=3*(0.5-rng.rand(x.shape[0]//5))
# # kr = KernelRidge(kernel="sigmoid",gamma=0.4)
# kr = GridSearchCV(KernelRidge(),
#                   param_grid={
#                       "kernel":["rbf", "laplacian", "polynomial", "sigmoid","linear"],
#                       "alpha": [1e0, 0.1, 1e-2, 1e-3],
#                       "gamma": np.logspace(-2, 2, 5)
#                   }
#
#                   )
# kr.fit(x,y)
# print(kr.best_params_,kr.best_score_)
# x_plot = np.linspace(0,5,100)
# t_kr = kr.predict(x_plot[...,None])
# plt.scatter(x,y)
# plt.plot(x_plot,t_kr,color="red")
# plt.show()
# if __name__ == '__main__':
#
#     print(np.logspace(-2, 2, 5))
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

rng = np.random.RandomState(0)

# X =5 * rng.rand(100, 1)
# y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

# kr = KernelRidge(kernel='sigmoid', gamma=0.4)
kr = GridSearchCV(KernelRidge(),
                  param_grid={"kernel": ["rbf", "laplacian", "polynomial", "sigmoid"],
                              "alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
kr.fit(X, y)
print(kr.best_score_, kr.best_params_)

X_plot = np.linspace(0, 5, 100)
y_kr = kr.predict(X_plot[:, None])

plt.scatter(X, y)
plt.plot(X_plot, y_kr, color="red")
plt.show()
