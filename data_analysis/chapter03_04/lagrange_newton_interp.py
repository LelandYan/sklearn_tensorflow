# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/22 19:45'

import pandas as pd
from scipy.interpolate import lagrange

# input_file = './data/catering_sale.xls'
# output_file = './tmp/sales.xls'
#
# data = pd.read_excel(input_file)
# # 过滤异常值 将其变为空值
# data[(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None


def programmer_1():
    inputfile = './data/catering_sale.xls'
    outputfile = './tmp/sales.xls'

    data = pd.read_excel(inputfile)

    data[(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None

    def ployinterp_column(index, df, k=5):
        y = df[list(range(index - k, index))
               + list(range(index + 1, index + 1 + k))]
        y = y[y.notnull()]
        return lagrange(y.index, list(y))(index)

    df = data[data[u'销量'].isnull()]

    index_list = df[u'销量'].index

    for index in index_list:
        data[[u'销量']][index] = ployinterp_column(index, data[u'销量'])

    data.to_excel(outputfile,index=False)

if __name__ == '__main__':
    programmer_1()