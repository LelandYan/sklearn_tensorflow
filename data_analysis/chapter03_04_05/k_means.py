# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/22 22:56'

import pandas as pd
if __name__ == '__main__':
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    inputfile = './data/consumption_data.xls'
    outputfile = './tmp/data_type.xls'

    k = 3
    iteration = 500
    data = pd.read_excel(inputfile, index_col='Id')
    data_zs = 1.0 * (data - data.mean()) / data.std()

    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    model.fit(data_zs)

    # 统计各个类别的数目
    r1 = pd.Series(model.labels_).value_counts()
    r2 = pd.DataFrame(model.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(data.columns) + [u'类别数目']
    print(r)

    # 详细输出每个样本对应的类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + [u'聚类类别']
    r.to_excel(outputfile)


    def density_plot(data, k):
        p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
        [p[i].set_ylabel(u'密度') for i in range(k)]
        plt.legend()
        return plt


    # 保存概率密度图
    pic_output = 'tmp/pd_'
    for i in range(k):
        density_plot(data[r[u'聚类类别'] == i],
                     k).savefig(u'%s%s.png' % (pic_output, i))