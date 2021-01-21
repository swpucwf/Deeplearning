import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.datasets import samples_generator
from sklearn import metrics, cluster

X, _ = samples_generator.make_blobs(n_samples=200, centers=2, cluster_std=0.60, random_state=0)
# X, y_true = samples_generator.make_moons(200, noise=0.05, random_state=0)
# X, y_true = samples_generator.make_circles(200, noise=0.05, random_state=0,factor=0.4)
# 混合高斯
# gmm = GaussianMixture(n_components=2)
# meanshift
# gmm = cluster.MeanShift()
# kmeans
# gmm = cluster.KMeans(4)
# 层次聚类
# gmm = cluster.AgglomerativeClustering(2)
# AP聚类，演讲算法
# gmm = cluster.AffinityPropagation()
# 谱聚类
# gmm = cluster.SpectralClustering(2,affinity="nearest_neighbors")
#密度聚类
gmm = cluster.DBSCAN()

labels = gmm.fit_predict(X)

# print(metrics.silhouette_score(X, labels))
# print(metrics.calinski_harabasz_score(X, labels))
# print(metrics.davies_bouldin_score(X, labels))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.show()
