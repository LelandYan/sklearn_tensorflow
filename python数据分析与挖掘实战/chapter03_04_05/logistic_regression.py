# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/22 22:13'
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR


filename = './data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()

# 建立随机逻辑回归模型，筛选变量
rlr = RLR()
rlr.fit(x,y)
# print(rlr.get_support())
# 获取特征筛选结果
print("通过随机逻辑回归模型筛选特征结束：")
print("有效特征:{} ".format(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix()

lr = LR()
lr.fit(x,y)
print("逻辑回归模型训练结束：")
print("模型的正确率为：",lr.score(x,y))