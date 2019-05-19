# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/9 7:25'
from scipy.sparse import dok_matrix,lil_matrix
import numpy as np
user_item = np.array([[1,0,0,1],[1,1,0,0],[1,0,0,1],[1,1,0,0]])
# train = dok_matrix(user_item.shape)
user_item = lil_matrix(user_item)
train = dok_matrix(user_item)
# print(np.asarray(train.nonzero()).T)
# print(user_item.rows)


