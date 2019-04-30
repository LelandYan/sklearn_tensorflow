# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/25 17:13'

import pandas as pd


def data_explore():
    datafile = './data/air_data.csv'
    resultfile = './tmp/explore.xls'

    data = pd.read_csv(datafile, encoding='utf-8')
    explore = data.describe(percentiles=[], include='all').T
    explore['null'] = len(data) - explore['count']  # 使用describe函数自动的计算非空数值

    explore = explore[['null', 'max', 'min']]
    explore.columns = ['空值数', '最大值', '最小值']
    explore.to_excel(resultfile)


def data_clean():
    datafile = './data/air_data.csv'
    cleanedfile = './tmp/data_cleaned.csv'

    data = pd.read_csv(datafile, encoding='utf8')
    data = data[data['SUM_YR_1'].notnull() * data['SUM_YR_2'].notnull()]

    index1 = data['SUM_YR_1'] != 0
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)

    data = data[index1 | index2 | index3]
    data.to_csv(cleanedfile)


def zscore_data():
    datafile = './data/zscoredata.xls'
    zscoredfile = './tmp/zscoreddata.xls'

    data = pd.read_excel(datafile)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data.columns = ['Z' + i for i in data.columns]
    data.to_excel(zscoredfile, index=False)


def KMeans_cluster():
    from sklearn.cluster import KMeans
    inputfile = './tmp/zscoreddata.xls'
    k = 5
    data = pd.read_excel(inputfile)

    kmodel = KMeans(n_clusters=k, n_jobs=4)
    kmodel.fit(data)
    r1 = pd.Series(kmodel.labels_).value_counts()
    r2 = pd.DataFrame(kmodel.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(data.columns) + ['类别数目']
    print(r)
    r = pd.concat([data, pd.Series(kmodel.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + ['聚类类别']
    print(r)
    # print(kmodel.cluster_centers_)  # 查看聚类中心
    # print(kmodel.labels_)  # 查看各样本对应的类别


def density_plot(data, title):
    import matplotlib.pyplot as plt
    # 正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 保存图像不能显示符号
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    for i in range(len(data.iloc[0])):
        (data.iloc[:, i].plot(kind='kde', label=data.columns[i], linewidth=2))
    plt.ylabel('密度')
    plt.xlabel('人数')
    plt.title(f'聚类类别{title}各属性的密度曲线')
    plt.legend()
    return plt


def density_plot(data):
    import matplotlib.pyplot as plt
    # 正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 保存图像不能显示符号
    plt.rcParams['axes.unicode_minus'] = False
    p = data.plot(kind='kde', subplots=True, sharex=False, linewidth=2)
    k = 3
    [p[i].set_ylabel('密度') for i in range(k)]
    plt.legend()
    return plt


if __name__ == '__main__':
    # data_explore()
    # data_clean()
    # zscore_data()
    KMeans_cluster()
