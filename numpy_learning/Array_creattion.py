# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/6/4 18:40'

import numpy as np
import os
import pandas as pd
# Create a new array of 2 * 2integers
print(np.empty([2,2]),int)

# empty_like函数
X = np.array([[1,2,3],[4,5,6]],np.int32)
print(np.empty_like(X))

# Create 2 3-D array with ones on the diagonal and zeros elsewhere
print(np.eye(3))
print(np.identity(3))

# Create a new array of 3*2 float numbers filled with ones
print(np.ones([3,2]),float)
x = np.arange(4,dtype=np.int64)
print(np.ones_like(x))

# Create a new array of 3 * 2 float number,filled with zeros
print(np.zeros((3,2),dtype=np.uint))
print(np.zeros_like(x))

# Create a new array of 2 * 5 uint8,filled with 6
print(np.full((2,5),6,dtype=np.uint))


# let list to array
X = [1,2]
print(np.asarray(X))


X = np.array([[ 0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9, 10, 11]])

print(np.diag(X))
print(np.diagflat([1,2,3,4]))


# 上三角矩阵
print(np.triu(np.arange(1,13).reshape(4,3),-1))

# 打印版本号
print(np.__version__)

x = np.ones([10,10,3])
out = np.reshape(x,[-1,150])
print(out.shape)
assert np.allclose(out,np.ones([10,10,3]).reshape([-1,150]))
x = np.array([[1,2,3],[4,5,6]])
print(x.flatten(order='F'))
print(x.ravel(order='F'))


out1 = x.flat[4]
out2 = np.ravel(x)[4]
print(out1,out2)

x = np.zeros((3,4,5))
out1 = np.swapaxes(x,1,0)
out2 = x.transpose([1,0,2])
print(out1.shape)

x = np.zeros((3,4))
# 进行维度的扩展
print(np.expand_dims(x,axis=0).shape)

# 进行维度降维
x = np.zeros((3,4,1))
print(np.squeeze(x).shape)


# 矩阵的合并
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7,8,9],[10,11,12]])
out1 = np.concatenate((x,y),1)
out2 = np.hstack((x,y))
print(out1)
print(out2)

x = np.array((1,2,3))
y = np.array((4,5,6))
out1 = np.concatenate((x,y),0)
out2 = np.vstack((x,y))
print(out2)

x = np.arange(16).reshape((4,4))
out1 = np.hsplit(x,2)
# print(x.shape)
print(out1)

# 实现数组中数据的重复
x = np.array([0,1,2])
out1 = np.tile(x,[2,2])
print(out1)
print(x.repeat(2))