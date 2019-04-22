# _*_ coding: utf-8 _*_
__author__ = 'Yxp'
__date__ = '2019/4/22 14:37'

import pandas as pd
# catering_sale = './data/catering_sale.xls'
# # 指定索引列
# data = pd.read_excel(catering_sale,index_col='日期')
# count stands for the not null number
# print(data.describe())
# print(data)
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# plt.figure(figsize=(12,8.5))
#
# # 画箱线图
# p = data.boxplot(return_type='dict')
# x = p['fliers'][0].get_xdata()
# y = p['fliers'][0].get_ydata()
# y.sort()
# for i in range(len(x)):
#     # 处理临界情况， i=0时
#     temp = y[i] - y[i - 1] if i != 0 else -78 / 3
#     # 添加注释, xy指定标注数据，xytext指定标注的位置（所以需要特殊处理）
#     plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / temp, y[i]))
# plt.show()

# data = data[(data['销量'] > 400) & (data['销量'] < 5000)]
# statistics = data.describe()
#
# statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']
# statistics.loc['var'] = statistics.loc['std'] / statistics.loc['mean']
# statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']
#
# print(statistics)

dish_profit = './data/catering_dish_profit.xls'
data = pd.read_excel(dish_profit, index_col='菜品名')
data = data['盈利'].copy()
data.sort_values(ascending=False)

plt.figure()
data.plot(kind='bar')
plt.ylabel('盈利（元）')
p = 1.0 * data.cumsum() / data.sum()
p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
plt.annotate(
    format(p[6], '.4%'),
    xy=(6, p[6]),
    xytext=(6 * 0.9, p[6] * 0.9),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.ylabel(u'盈利（比例）')
plt.show()