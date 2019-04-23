# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/23 7:33'

import pandas as pd
from apriori import *

inputfile = './data/menu_orders.xls'
outputfile = './tmp/apriori_rules.xls'

data = pd.read_excel(inputfile,header=None)
print("\n转换原始数据至0-1矩阵...")
ct = lambda x:pd.Series(1,index=x[pd.notnull(x)])
b = map(ct,data.as_matrix())
data = pd.DataFrame(list(b)).fillna(0)
print("\n转化完成")

support = 0.2
confidence = 0.5
ms = "---"
file_rule(data,support,confidence,ms).to_excel(outputfile)