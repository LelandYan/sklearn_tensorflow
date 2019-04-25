# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/25 21:17'

import pandas as pd


def divide_event():
    threshold = pd.Timedelta(minutes=4)  # 阀值为4分钟
    inputfile = './data/water_heater.xls'
    outputfile = './tmp/dividsequence.xls'

    data = pd.read_excel(inputfile)
    data['发生时间'] = pd.to_datetime(data['发生时间'], format='%Y%m%d%H%M%S')
    d = data["发生时间"].diff() > threshold  # 相邻时间作差分，大于threshold
    data['事件编号'] = d.cumsum() + 1

    data.to_excel(outputfile)

def threshold_optimization():
    import numpy as np
    import pandas as pd
    inputfile = "data/water_heater.xls"
    # 使用之后四个点的平均斜率
    n = 4

    # 专家阈值
    threshold = pd.Timedelta(minutes=5)
    data = pd.read_excel(inputfile)
    data[u"发生时间"] = pd.to_datetime(data[u"发生时间"], format="%Y%m%d%H%M%S")
    data = data[data[u"水流量"] > 0]

    # 定义阈值列
    dt = [pd.Timedelta(minutes=i) for i in np.arange(1, 9, 0.25)]
    h = pd.DataFrame(dt, columns=[u"阈值"])

    def event_num(ts):
        d = data[u"发生时间"].diff() > ts
        # 返回事件数
        return d.sum() + 1

    # 计算每个阈值对应的事件数
    h[u"事件数"] = h[u"阈值"].apply(event_num)
    # 计算每两个相邻点对应的斜率
    h[u"斜率"] = h[u"事件数"].diff() / 0.25
    # 采用后n个的斜率绝对值平均作为斜率指标
    h[u"斜率指标"] = pd.Series.rolling(h[u"斜率"].abs(), n).mean()
    ts = h[u"阈值"][h[u"斜率指标"].idxmin() - n]

    if ts > threshold:
        ts = pd.Timedelta(minutes=4)

    print(ts)


if __name__ == '__main__':
    # divide_event()
    threshold_optimization()