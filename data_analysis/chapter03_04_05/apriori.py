# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/23 12:27'
import pandas as pd


def connect_string(x, ms):
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r


def file_rule(d, support, confidence, ms='---'):
    # 定义输出结果
    result = pd.DataFrame(index=['support', 'confidence'])
    # 支持度序列
    support_series = 1.0 * d.sum() / len(d)
    # 支持度第一次筛选
    column = list(support_series[support_series > support].index)
    k = 0
    while len(column) > 1:
        k = k + 1
        print(u'\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print(u'数目：%s...' % len(column))

        # 新一批支持度的计算函数
        def sf(i):
            return d[i].prod(axis=1, numeric_only=True)

        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(
            list(map(sf, column)), index=[ms.join(i) for i in column]).T

        # 计算连接后的支持度
        support_series_2 = 1.0 * \
                           d_2[[ms.join(i) for i in column]].sum() / len(d)
        column = list(
            support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        cofidence_series = pd.Series(
            index=[ms.join(i) for i in column2])  # 定义置信度序列
        # 计算置信度序列
        for i in column2:
            cofidence_series[ms.join(i)] = support_series[ms.join(
                sorted(i))] / support_series[ms.join(i[:len(i) - 1])]
        # 置信度筛选
        for i in cofidence_series[cofidence_series > confidence].index:
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(
        ['confidence', 'support'], ascending=False)  # 结果整理，输出
    print(u'\n结果为：')
    print(result)

    return result
