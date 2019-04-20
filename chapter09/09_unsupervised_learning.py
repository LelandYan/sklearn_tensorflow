# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/18 18:31'

from sklearn.datasets import load_iris
import matplotlib as mpl
import matplotlib.pyplot as plt

# data = load_iris()
# X = data.data
# y = data.target

# plt.subplot(121)
# plt.plot(X[y == 0, 2], X[y == 0, 3], "yo", label="Iris-Setosa")
# plt.plot(X[y == 1, 2], X[y == 1, 3], "bs", label="Iris-Versicolor")
# plt.plot(X[y == 2, 2], X[y == 2, 3], "g^", label="Iris-Virginica")
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(fontsize=12)
#
# plt.subplot(122)
# plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
# plt.xlabel("Petal length", fontsize=14)
# plt.tick_params(labelleft=False)
#
# plt.show()

from sklearn.mixture import GaussianMixture
import numpy as np

# y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
# mapping = np.array([2, 0, 1])
# y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])
# plt.plot(X[y_pred == 0, 2], X[y_pred == 0, 3], "yo", label="Cluster 1")
# plt.plot(X[y_pred == 1, 2], X[y_pred == 1, 3], "bs", label="Cluster 2")
# plt.plot(X[y_pred == 2, 2], X[y_pred == 2, 3], "g^", label="Cluster 3")
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="upper left", fontsize=12)
# # plt.show()
# print(np.sum(y_pred == y) / len(y_pred))

# K-Means
from sklearn.datasets import make_blobs

# blob_centers = np.array([
#     [0.2, 2.3],
#     [-1.5, 2.3],
#     [-2.8, 1.8],
#     [-2.8, 2.8],
#     [-2.8, 1.3]
# ])
# blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
# X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)


def plot_cluster(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14, rotation=0)


# plt.figure(figsize=(8,4))
# plot_cluster(X)
# plt.show()

# from sklearn.cluster import KMeans
# k = 5
# kmeans = KMeans(n_clusters=k,random_state=42)
# y_pred = kmeans.fit_predict(X)
# print(kmeans.cluster_centers_)
# print(y_pred == kmeans.labels_)

# Decision Boundaries
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundary(clusterer, X, resolution=1000, show_centroids=True,
                           show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


# plt.figure(figsize=(8, 4))
# plot_decision_boundary(kmeans, X)
# plt.show()
#
# kmeans_iter1 = KMeans(n_clusters=5,init='random',n_init=1,algorithm='full',max_iter=1,random_state=1)
# kmeans_iter2 = KMeans(n_clusters=5,init='random',n_init=1,algorithm='full',max_iter=2,random_state=1)
# kmeans_iter3 = KMeans(n_clusters=5,init='random',n_init=1,algorithm='full',max_iter=3,random_state=1)
#
# kmeans_iter1.fit(X)
# kmeans_iter2.fit(X)
# kmeans_iter3.fit(X)
# plt.figure(figsize=(10, 8))
#
# plt.subplot(321)
# plot_data(X)
# plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
# plt.ylabel("$x_2$", fontsize=14, rotation=0)
# plt.tick_params(labelbottom=False)
# plt.title("Update the centroids (initially randomly)", fontsize=14)
#
# plt.subplot(322)
# plot_decision_boundary(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
# plt.title("Label the instances", fontsize=14)
#
# plt.subplot(323)
# plot_decision_boundary(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
# plot_centroids(kmeans_iter2.cluster_centers_)
#
# plt.subplot(324)
# plot_decision_boundary(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)
#
# plt.subplot(325)
# plot_decision_boundary(kmeans_iter2, X, show_centroids=False)
# plot_centroids(kmeans_iter3.cluster_centers_)
#
# plt.subplot(326)
# plot_decision_boundary(kmeans_iter3, X, show_ylabels=False)

def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundary(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundary(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


# kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,
#                          algorithm="full", random_state=11)
# kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,
#                          algorithm="full", random_state=19)
#
# plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
#                           "Solution 1", "Solution 2 (with a different random init)")
#
# plt.show()

from sklearn.cluster import MiniBatchKMeans
# minibatch_kmeans = MiniBatchKMeans(n_clusters=5,random_state=42)
# minibatch_kmeans.fit(X)

from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# 导入数据
# mnist = loadmat("mnist-original.mat")
# X = mnist["data"].T
# y = mnist["label"][0]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# filename = 'my_mnist.data'
# X_mm = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
# X_mm[:] = X_train
# minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
# minibatch_kmeans.fit(X_mm)
#
#
# def load_next_batch(batch_size):
#     return X[np.random.choice(len(X), batch_size, replace=False)]


# np.random.seed(42)
# k = 5
# n_init = 10
# n_iterations = 100
# batch_size = 100
# init_size = 500  # more data for K-Means++ initialization
# evaluate_on_last_n_iters = 10
#
# best_kmeans = None

# for init in range(n_init):
#     minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
#     X_init = load_next_batch(init_size)
#     minibatch_kmeans.partial_fit(X_init)
#
#     minibatch_kmeans.sum_inertia_ = 0
#     for iteration in range(n_iterations):
#         X_batch = load_next_batch(batch_size)
#         minibatch_kmeans.partial_fit(X_batch)
#         if iteration >= n_iterations - evaluate_on_last_n_iters:
#             minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_
#
#     if (best_kmeans is None or
#         minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):
#         best_kmeans = minibatch_kmeans
#
# print(best_kmeans.score(X))
# print(best_kmeans.inertia_)


from timeit import timeit
from sklearn.cluster import KMeans
# times = np.empty((100,2))
# inertias = np.empty((100,2))
# for k in range(1,101):
#     kmeans = KMeans(n_clusters=k,random_state=42)
#     minibatch_kmeans = MiniBatchKMeans(n_clusters=k,random_state=42)
#     print("\r{}/{}".format(k, 100), end="")
#     times[k - 1, 0] = timeit("kmeans.fit(X)", number=10, globals=globals())
#     times[k - 1, 1] = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
#     inertias[k - 1, 0] = kmeans.inertia_
#     inertias[k - 1, 1] = minibatch_kmeans.inertia_
#
# plt.figure(figsize=(10,4))
#
# plt.subplot(121)
# plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
# plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
# plt.xlabel("$k$", fontsize=16)
# #plt.ylabel("Inertia", fontsize=14)
# plt.title("Inertia", fontsize=14)
# plt.legend(fontsize=14)
# plt.axis([1, 100, 0, 100])
#
# plt.subplot(122)
# plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
# plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
# plt.xlabel("$k$", fontsize=16)
# #plt.ylabel("Training time (seconds)", fontsize=14)
# plt.title("Training time (seconds)", fontsize=14)
# plt.axis([1, 100, 0, 6])
# #plt.legend(fontsize=14)
#
# plt.show()


# kmeans_k3 = KMeans(n_clusters=3,random_state=42)
# kmeans_k4 = KMeans(n_clusters=8,random_state=42)
#
# plot_clusterer_comparison(kmeans_k3,kmeans_k4,X,'$k=3$','$k=4$')
# plt.show()


# kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
#                 for k in range(1, 10)]
# inertias = [model.inertia_ for model in kmeans_per_k]
#
# plt.figure(figsize=(8, 3.5))
# plt.plot(range(1, 10), inertias, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Inertia", fontsize=14)
# plt.annotate('Elbow',
#              xy=(4, inertias[3]),
#              xytext=(0.55, 0.55),
#              textcoords='figure fraction',
#              fontsize=16,
#              arrowprops=dict(facecolor='black', shrink=0.1)
#             )
# plt.axis([1, 8.5, 0, 1300])
# plt.show()
#
#
# from sklearn.metrics import silhouette_score
# silhouette_scores = [silhouette_score(X, model.labels_)
#                      for model in kmeans_per_k[1:]]

from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedFormatter, FixedLocator

from sklearn.datasets import load_digits
#
X_digits, y_digits = load_digits(return_X_y=True)
from sklearn.model_selection import train_test_split
#
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
from sklearn.linear_model import LogisticRegression
#
# log_reg =LogisticRegression(multi_class='ovr',solver='lbfgs',random_state=42)
# log_reg.fit(X_train,y_train)
# print(log_reg.score(X_test,y_test))
#
from sklearn.pipeline import Pipeline
# pipeline = Pipeline([
#     ("kmeans", KMeans(n_clusters=50, random_state=42)),
#     ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)),
# ])
# pipeline.fit(X_train, y_train)
# print(pipeline.score(X_test, y_test))
#
from sklearn.model_selection import GridSearchCV
# param_grid = dict(kmeans__n_clusters=range(2,100))
# grid_clf = GridSearchCV(pipeline,param_grid=param_grid,cv=3,verbose=2)
# grid_clf.fit(X_train,y_train)
# print(grid_clf.best_params_)
# print(grid_clf.best_estimator_.score(X_test,y_test))

#取前50个样本进行训练
# n_labeled = 50
# log_reg = LogisticRegression(multi_class='ovr',solver='lbfgs',random_state=42)
# log_reg.fit(X_train[:n_labeled],y_train[:n_labeled])
# res = log_reg.score(X_test,y_test)
# print("取前50个原本训练：",res,X_train.shape)
#
# # 聚类的中心数 --这里也相当于将10个类别进行分类成了50个
# k = 50
# kmeans = KMeans(n_clusters=k,random_state=42)
# X_digits_dist = kmeans.fit_transform(X_train)
# print("聚类后数据的形状: ",X_digits_dist.shape)
# # 这里选择最具有代表性的(最靠近质心的图像)50个，也就是聚类的中心点
# representative_digit_idx = np.argmin(X_digits_dist,axis=0)
# print("选取最靠近数据质心的索引：",representative_digit_idx.shape)
# X_representative_digits = X_train[representative_digit_idx]
# print("提取后的数据：",X_representative_digits.shape)
#
# # 对聚类后的结果进行可视化
# # plt.figure(figsize=(8, 2))
# # for index, X_representative_digit in enumerate(X_representative_digits):
# #     plt.subplot(k // 10, 10, index + 1)
# #     plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
# #     plt.axis('off')
# # plt.show()
#
# # 聚类后的图像所对应的标签
# y_representative_digits = np.array([
#     4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
#     5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
#     1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
#     6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
#     4, 2, 9, 4, 7, 6, 2, 3, 1, 1])
# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
# log_reg.fit(X_representative_digits, y_representative_digits)
# res = log_reg.score(X_test, y_test)
# print("使用聚类的质心的50张图像：",res)
#
# # label 数据集 进行标签传播算法
# y_train_propagated = np.empty(len(X_train), dtype=np.int32)
# for i in range(k):
#     # 取聚类后标签为i的样本
#     y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
# log_reg.fit(X_train, y_train_propagated)
# print("使用聚类后的50张图片进行标签扩展到整个数据集：",log_reg.score(X_test,y_test))
#
#
# # 标签传播法传播到20%
# percentile_closest = 20
# # 按聚类后的标签对聚类后的数据处理
# X_cluster_dist = X_digits_dist[np.arange(len(X_train)),kmeans.labels_]
#
# for i in range(k):
#     # 取出标签为i的聚类后的X_cluster_dist数据索引
#     in_cluster = (kmeans.labels_ == i)
#     cluster_dist = X_cluster_dist[in_cluster]
#     # 求数据中20%的分位数
#     cutoff_distance = np.percentile(cluster_dist,percentile_closest)
#     above_cutoff = (X_cluster_dist > cutoff_distance)
#     X_cluster_dist[in_cluster & above_cutoff] = -1
#
# partially_propagated = (X_cluster_dist != -1)
# X_train_partially_propagated = X_train[partially_propagated]
# y_train_partially_propagated = y_train_propagated[partially_propagated]
#
# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
# log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
# print("X_train经过聚类处理的数据:",X_train_partially_propagated.shape)
# res = log_reg.score(X_test, y_test)
# print("取距离聚类后的质心为20%的数据：",res)

# DBSCAN
from sklearn.datasets import make_moons
#
X,y = make_moons(n_samples=1000,noise=0.05,random_state=42)
#
# from sklearn.cluster import DBSCAN
#
# dbscan = DBSCAN(eps=0.05,min_samples=5)
# dbscan.fit(X)
#
# dbscan2 = DBSCAN(eps=0.2)
# dbscan2.fit(X)
#
# def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
#     core_mask = np.zeros_like(dbscan.labels_,dtype=bool)
#     core_mask[dbscan.core_sample_indices_] = True
#     anomalies_mask = dbscan.labels_ == -1
#     non_core_mask = ~(core_mask | anomalies_mask)
#
#     cores = dbscan.components_
#     anomalies = X[anomalies_mask]
#     non_cores = X[non_core_mask]
#
#     plt.scatter(cores[:, 0], cores[:, 1],
#                 c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
#     plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
#     plt.scatter(anomalies[:, 0], anomalies[:, 1],
#                 c="r", marker="x", s=100)
#     # plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
#     if show_xlabels:
#         plt.xlabel("$x_1$", fontsize=14)
#     else:
#         plt.tick_params(labelbottom=False)
#     if show_ylabels:
#         plt.ylabel("$x_2$", fontsize=14, rotation=0)
#     else:
#         plt.tick_params(labelleft=False)
#     plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)
#
# plt.figure(figsize=(9, 3.2))
#
# plt.subplot(121)
# plot_dbscan(dbscan, X, size=100)
#
# plt.subplot(122)
# plot_dbscan(dbscan2, X, size=600, show_ylabels=False)
#
# # plt.show()
# dbscan = dbscan2
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=50)
# knn.fit(dbscan.components_,dbscan.labels_[dbscan.core_sample_indices_])
# X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
# knn.predict(X_new)


# Spectral Clustering
# from sklearn.cluster import SpectralClustering
# scl = SpectralClustering(n_clusters=2,gamma=100,random_state=42)
# scl.fit(X)
# sc2 = SpectralClustering(n_clusters=2,gamma=1,random_state=42)
# print(np.percentile(scl.affinity_matrix_,95))


def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=size, c='gray', cmap="Paired", alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=10, c=sc.labels_, cmap="Paired")

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)


# from sklearn.cluster import AgglomerativeClustering
# X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
# agg = AgglomerativeClustering(linkage="complete").fit(X)
# def learned_parameters(estimator):
#     return [attrib for attrib in dir(estimator)
#             if attrib.endswith("_") and not attrib.startswith("_")]
# agg.children_

X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
print(X1.shape)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
gm = GaussianMixture(n_components=3,n_init=10,random_state=42)
gm.fit(X)
print(gm.weights_)
print(gm.means_)
print(gm.covariances_)
# 判断算法时候收敛
print(gm.converged_)
# 需要多少次迭代
print(gm.n_iter_)

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
np.round(bgm.weights_, 2)

from scipy.stats import norm
xx = np.linspace(-6, 4, 101)
ss = np.linspace(1, 2, 101)
XX, SS = np.meshgrid(xx, ss)
ZZ = 2 * norm.pdf(XX - 1.0, 0, SS) + norm.pdf(XX + 4.0, 0, SS)
ZZ = ZZ / ZZ.sum(axis=1) / (xx[1] - xx[0])
from matplotlib.patches import Polygon

plt.figure(figsize=(8, 4.5))

x_idx = 85
s_idx = 30

plt.subplot(221)
plt.contourf(XX, SS, ZZ, cmap="GnBu")
plt.plot([-6, 4], [ss[s_idx], ss[s_idx]], "k-", linewidth=2)
plt.plot([xx[x_idx], xx[x_idx]], [1, 2], "b-", linewidth=2)
plt.xlabel(r"$x$")
plt.ylabel(r"$\theta$", fontsize=14, rotation=0)
plt.title(r"Model $f(x; \theta)$", fontsize=14)

plt.subplot(222)
plt.plot(ss, ZZ[:, x_idx], "b-")
max_idx = np.argmax(ZZ[:, x_idx])
max_val = np.max(ZZ[:, x_idx])
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [0, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.text(1.01, max_val + 0.005, r"$\hat{L}$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, 0.055, r"$\hat{\theta}$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, max_val - 0.012, r"$Max$", fontsize=12)
plt.axis([1, 2, 0.05, 0.15])
plt.xlabel(r"$\theta$", fontsize=14)
plt.grid(True)
plt.text(1.99, 0.135, r"$=f(x=2.5; \theta)$", fontsize=14, ha="right")
plt.title(r"Likelihood function $\mathcal{L}(\theta|x=2.5)$", fontsize=14)

plt.subplot(223)
plt.plot(xx, ZZ[s_idx], "k-")
plt.axis([-6, 4, 0, 0.25])
plt.xlabel(r"$x$", fontsize=14)
plt.grid(True)
plt.title(r"PDF $f(x; \theta=1.3)$", fontsize=14)
verts = [(xx[41], 0)] + list(zip(xx[41:81], ZZ[s_idx, 41:81])) + [(xx[80], 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
plt.gca().add_patch(poly)

plt.subplot(224)
plt.plot(ss, np.log(ZZ[:, x_idx]), "b-")
max_idx = np.argmax(np.log(ZZ[:, x_idx]))
max_val = np.max(np.log(ZZ[:, x_idx]))
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [-5, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.axis([1, 2, -2.4, -2])
plt.xlabel(r"$\theta$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, max_val - 0.05, r"$Max$", fontsize=12)
plt.text(ss[max_idx]+ 0.01, -2.39, r"$\hat{\theta}$", fontsize=14)
plt.text(1.01, max_val + 0.02, r"$\log \, \hat{L}$", fontsize=14)
plt.grid(True)
plt.title(r"$\log \, \mathcal{L}(\theta|x=2.5)$", fontsize=14)

plt.show()