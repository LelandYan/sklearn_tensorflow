# _*_ coding: utf-8 _*_
__author__ = 'Yxp'
__date__ = '2019/4/22 14:37'

import pandas as pd
catering_sale = './data/catering_sale.xls'
# 指定索引列
data = pd.read_excel(catering_sale,index_col='日期')
# count stands for the not null number
# print(data.describe())
# print(data)
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(12,8.5))

# 画箱线图
p = data.boxplot(return_type='dict')
x = p['fliers'][0].get_xdata()
y = p['fliers'][0].get_ydata()
y.sort()
for i in range(len(x)):
    # 处理临界情况， i=0时
    temp = y[i] - y[i - 1] if i != 0 else -78 / 3
    # 添加注释, xy指定标注数据，xytext指定标注的位置（所以需要特殊处理）
    plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / temp, y[i]))
plt.show()
