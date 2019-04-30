# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/22 20:27'

import pandas as pd

# datafile = './data/discretization_data.xls'
# data = pd.read_excel(datafile)
# data = data['肝气郁结证型系数'].copy()
# k = 4



def cluster_plot(d, k):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')
    plt.ylim(-0.5, k - 0.5)
    return plt

if __name__ == '__main__':
    datafile = './data/discretization_data.xls'
    data = pd.read_excel(datafile)
    data = data[u'肝气郁结证型系数'].copy()
    k = 4

    # 方法一， 直接对数组进行分类
    d1 = pd.cut(data, k, labels=range(k))

    # 方法二， 等频率离散化
    w = [1.0 * i / k for i in range(k + 1)]
    # percentiles表示特定百分位数，同四分位数
    w = data.describe(percentiles=w)[4:4 + k + 1]
    w[0] = w[0] * (1 - 1e-10)
    d2 = pd.cut(data, w, labels=range(k))

    from sklearn.cluster import KMeans

    # 　方法三，使用Kmeans
    kmodel = KMeans(n_clusters=k, n_jobs=4)

    kmodel.fit(data.values.reshape(len(data), 1))
    # 输出聚类中心，并且排序

    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)

    # 相邻两项求中点，作为边界点
    w = pd.DataFrame.rolling(c, 2).mean().iloc[1:]
    # 加上首末边界点
    w = [0] + list(w[0]) + [data.max()]
    d3 = pd.cut(data, w, labels=range(k))

    # cluster_plot(d1, k).show()
    # cluster_plot(d2, k).show()
    # cluster_plot(d3, k).show()

#
# # 等宽离散化
# d1 = pd.cut(data,k,labels=range(k))
#
# # 等频离散化
# w = [1.0*i/k for i in range(k+1)]
# w = data.describe(percentiles=w)[4:4+k+1] # 使用describe函数计算分位数
# w[0] = w[0] * (1-1e-10)
# d2 = pd.cut(data,w,labels=range(k))
#
# from sklearn.cluster import KMeans
# kmodel = KMeans(n_clusters=k,n_jobs=-1)
# kmodel.fit(data.reshape((len(data),1)))
# c = pd.DataFrame(kmodel.cluster_centers_).sort(0)
# w = pd.rolling_mean(c,2).iloc[1:]
# w = [0] + list(w[0]) + data.max()
# d3 = pd.cut(data,w,labels=range(k))
#
#
# def cluster_plot(d, k):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 3))
#     for j in range(0, k):
#         plt.plot(data[d == j], [j for i in d[d == j]], 'o')
#     plt.ylim(-0.5, k - 0.5)
#     return plt
#
#
# cluster_plot(d1, k).show()
# cluster_plot(d2, k).show()
# cluster_plot(d3, k).show()