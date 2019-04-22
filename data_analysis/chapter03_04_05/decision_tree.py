# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/22 22:47'

import pandas as pd

filename = './data/sales_data.xls'
data = pd.read_excel(filename,index_col='序号')

data[data=='好'] = 1
data[data=='是'] = 1
data[data=='高'] = 1
data[data != 1] = -1

x = data.iloc[:,:3].as_matrix()
y = data.iloc[:,3].as_matrix()

from sklearn.tree import DecisionTreeClassifier as DTC

dtc = DTC(criterion='entropy')
dtc.fit(x.y)

