# _*_ coding: utf-8 _*_
__author__ = 'Yxp'
__date__ = '2019/4/21 12:55'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# np.random.seed(4)
# m = 60
# w1, w2 = 0.1, 0.3
# noise = 0.1
#
# # 构造数据
# angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
# X = np.empty((m, 3))
# X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
# X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
# X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
#
# X_centered = X-X.mean(axis=0)
# U,s,Vt = np.linalg.svd(X_centered)
# c1 = Vt.T[:,0]
# c2 = Vt.T[:,1]

# m,n = X.shape
#
# S = np.zeros(X_centered.shape)

# pca
from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X2D = pca.fit_transform(X)
# X3D_inv = pca.inverse_transform(X2D)
# # compute the reconstruction error
# # print(np.mean(np.sum(np.square(X3D_inv-X)),axis=1))
# # the explained variance ratio
# print(pca.explained_variance_ratio_)
# print(pca.n_components_)

from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import make_swiss_roll
# X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
from sklearn.manifold import LocallyLinearEmbedding

# lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
# X_reduced = lle.fit_transform(X)
# plt.title("Unrolled swiss roll using LLE", fontsize=14)
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18)
# plt.axis([-0.065, 0.055, -0.1, 0.12])
# plt.grid(True)

# plt.show()
# from sklearn.manifold import MDS
#
# mds = MDS(n_components=2, random_state=42)
# X_reduced_mds = mds.fit_transform(X)
#
# from sklearn.manifold import Isomap
#
# isomap = Isomap(n_components=2)
# X_reduced_isomap = isomap.fit_transform(X)
#
from sklearn.manifold import TSNE
#
# tsne = TSNE(n_components=2, random_state=42)
# X_reduced_tsne = tsne.fit_transform(X)
#
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# lda = LinearDiscriminantAnalysis(n_components=2)

from scipy.io import loadmat

mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]

np.random.seed(42)
m = 10000
# 随机取10000个数据
idx = np.random.permutation(60000)[:m]
X = X[idx]
y = y[idx]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,random_state=42)
X_reduced = tsne.fit_transform(X)
# plt.figure(figsize=(3,10))
# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y,cmap='jet')
# plt.colorbar()
# plt.show()
plt.figure(figsize=(9,9))
cmap = mpl.cm.get_cmap('jet')
