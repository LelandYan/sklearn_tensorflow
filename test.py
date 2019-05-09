import numpy as np
import tensorflow as tf
import math
from scipy.special import gamma


# def gamma(x):
#    q=1
#    while True:
#        if x-1>=1:
#            x=x-1
#            q=q*x
#        else:
#            break
#    return q
# r=tf.Variable(1)
# def cond(x,r):
#   x>1
# def body(x,r):
#   x=x-1
#  r=r*x
# return x,r
# def mma(x,r):
#   x,r=tf.while_loop(cond, body, [x, r])
#  return r
# 定义伽马函数
def pearson(j, k, l, x):
    a = [j, k, l]
    # j k l 分别表示α β a0三个参数
    s = a[1] ** a[0]
    s = s / gamma(a[0])
    y = (x - a[2]) ** (a[0] - 1)
    s = s * y
    b = -a[1] * (x - a[2])
    s = s * math.exp(b)
    return s


# 完整定义皮尔逊Ⅲ型曲线
a = []
for j in np.arange(1, 11, 1):
    for k in np.arange(2, 3, .1):
        for l in np.arange(.1, 1, .1):
            for i in np.arange(1, 21, 1):
                a.append(pearson(j, k, l, i))
b = []
for x in range(900):
    b.append(a[20 * x:20 + 20 * x])
# b用来装纵坐标对应的值，列表套列表
c = {}
for i in range(900):
    c[i] = b[i]
# c为b中的每个列表加上标签，c是一个字典
d = []
for j in np.arange(1, 2, .1):
    for k in np.arange(2, 3, .1):
        for l in np.arange(.1, 1, .1):
            d.append([j, k, l])
# d用来装α β a0三个参数，列表套列表
e = {}
for i in range(360):
    e[i] = d[i]
# e为d中的每个列表加上标签，e也是一个字典
u = np.array(b)[:850, :]
q = np.array(d)[:850, :]
o = np.array(b)[850:, :]
p = np.array(d)[850:, :]

x = tf.placeholder(tf.float32, [1, 20])
y = tf.placeholder(tf.float32, [1, 3])
# z=tf.placeholder(tf.float32,[1,20])
# z中盛放预测结果
WeightL1 = tf.Variable(tf.random_normal([20, 4]))
BasisL1 = tf.Variable(tf.zeros([1, 4]))
L1 = tf.matmul(x, WeightL1) + BasisL1
taL1 = tf.nn.tanh(L1)

WeightL3 = tf.Variable(tf.random_normal([4, 10]))
BasisL3 = tf.Variable(tf.zeros([1, 10]))
L3 = tf.matmul(taL1, WeightL3) + BasisL3
taL3 = tf.nn.tanh(L3)

WeightL2 = tf.Variable(tf.random_normal([10, 3]))
BasisL2 = tf.Variable(tf.zeros([1, 3]))
L2 = tf.matmul(taL3, WeightL2) + BasisL2
taL2 = tf.nn.tanh(L2)
# 简单神经网络构建

init = tf.global_variables_initializer()
# 初始化参数
# opti=tf.train.GradientDescentOptimizer(0.3).minimize(tf.reduce_mean(tf.square(y-taL2)))
# 梯度下降法修正W和b两个参数
w = tf.to_float(taL2)[0][0]
i = tf.to_float(taL2)[0][1]
e = tf.to_float(taL2)[0][2]
# 将taL2中的元素取出来，方便后面使用
F = []
for x in range(20):
    a = tf.py_func(pearson, [w, i, e, x], tf.float64)
    F.append(a)
    # print(pearson(u,i,o,x))
# F为预测曲线中取点的纵坐标
opti = tf.train.GradientDescentOptimizer(0.3).minimize(tf.reduce_mean(tf.square(F - x)))
q = 0
for i in range(20):
    m = (tf.to_float(x)[0][i] - tf.to_float(F)[0][i]) / tf.to_float(x)[0][i]
    q = q + abs(m)
result = q / 20
# result表示20个点的平均相对误差，也就是对应曲线预测的误差值

# some = 0
# for i in range(3):
#    some = some + abs(tf.to_float(taL2)[0][i] - tf.to_float(y)[0][i]) / tf.to_float(y)[0][i]
# somes = 0
# result=tf.cond(some < 0.9, lambda: tf.add(somes, 1),lambda: tf.add(somes, 0))

with tf.Session() as sess:
    sess.run(init)
    x_in = sess.run(tf.convert_to_tensor(u))
    y_in = sess.run(tf.convert_to_tensor(q))
    x_innn = sess.run(tf.convert_to_tensor(o))
    y_innn = sess.run(tf.convert_to_tensor(p))
    for i in range(10):
        for i in range(850):
            sess.run(opti, feed_dict={x: x_in[i].reshape(1, 20), y: y_in[i].reshape(1, 3)})
        for i in range(50):
            data = sess.run(result, feed_dict={x: x_innn[i].reshape(1, 20), y: y_innn[i].reshape(1, 3)})
            print(data)
