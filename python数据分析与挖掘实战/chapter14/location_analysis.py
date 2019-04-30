# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/28 16:06'
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
import pandas as pd
def programmer_2():
    standardizedfile = "data/standardized.xls"
    data = pd.read_excel(standardizedfile, index_col=u"基站编号")
    Z = linkage(data, method="ward", metric="euclidean")
    P = dendrogram(Z, 0)
    plt.show()
def programmer_3():

    standardizedfile = "data/standardized.xls"
    k = 3
    data = pd.read_excel(standardizedfile, index_col=u"基站编号")

    # 层次聚类
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    model.fit(data)

    # 详细输入原始数据及对应类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + [u"聚类类别"]

    # 绘制聚类图，并且用不同样式进行画图
    style = ["ro-", "go-", "bo-"]
    xlabels = [u"工作日人均停留时间", u"凌晨人均停留时间", u"周末人均停留时间", u"日均人流量"]
    pic_output = "tmp/type_"

    for i in range(k):
        plt.figure()
        tmp = r[r[u"聚类类别"] == i].iloc[:, :4]
        for j in range(len(tmp)):
            plt.plot(range(1, 5), tmp.iloc[j], style[i])

        plt.xticks(range(1, 5), xlabels, rotation=20)

        plt.title(u"商圈类别%s" % (i + 1))
        # 调整底部
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(u"%s%s.png" % (pic_output, i + 1))
        plt.show()
if __name__ == '__main__':
    programmer_3()