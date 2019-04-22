# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/22 22:56'

import pandas as pd
if __name__ == '__main__':

    inputfile = './data/consumption_data.xls'
    outputfile = './tmp/data_type.xls'

    k = 3
    iterations = 500
    data = pd.read_excel(inputfile,index_col='Id')
    data_zs = 1.0 * (data - data-data.mean())/data.std()
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k,n_jobs=4,max_iter=iterations)
    model.fit(data_zs)

    # 统计各个类别的数目
    r1 = pd.Series(model.labels_).value_counts()
    # 找出聚类中心
    r2 = pd.DataFrame(model.cluster_centers_)
    print(model.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    print(r)
    r.columns = list(data.columns) + [u'类别数目']
    # print(r)

    # 详细输出每个样本对应的类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + [u'聚类类别']
    # print(r)
    r.to_excel(outputfile)