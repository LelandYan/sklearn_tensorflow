# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/23 7:56'

import numpy as np
import pandas as pd
if __name__ == '__main__':

    inputfile = "./data/consumption_data.xls"
    # 聚类的类别
    k = 3
    # 聚类的最大循环次数
    iteration = 500
    # 离散点阀值
    threshold = 2
    data = pd.read_excel(inputfile,index_col='Id')
    # 数据标准化
    data_zs = 1.0 * (data - data.mean()) / data.std()

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)
    model.fit(data_zs)

    r = pd.concat(
        [data_zs, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + [u'聚类类别']

    norm = []
    # 逐一处理
    for i in range(k):
        norm_tmp = r[['R','F','M']][r['聚类类别']==i] - model.cluster_centers_[i]
        norm_tmp = norm_tmp.apply(np.linalg.norm,axis=1)
        norm.append(norm_tmp/norm_tmp.median())
    norm = pd.concat(norm)

    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    norm[norm <= threshold].plot(style='go')
    discrete_points = norm[norm > threshold]
    discrete_points.plot(style='ro')
    for i in range(len(discrete_points)):
        _id = discrete_points.index[i]
        n = discrete_points.iloc[i]
        plt.annotate('(%s, %0.2f)' % (_id, n), xy=(_id, n), xytext=(_id, n))

    plt.xlabel(u'编号')
    plt.ylabel(u'相对距离')
    plt.show()
