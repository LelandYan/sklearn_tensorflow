# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/24 20:35'

from random import shuffle

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from scipy.interpolate import lagrange
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def Lagrange_interpolation():
    inputfile = "data/missing_data.xls"
    outputfile = 'tmp/missing_data_processed.xls'

    data = pd.read_excel(inputfile, header=None)

    # 自定义列向量插值函数
    # s为列向量，n为被插值的位置，k为取前后的数据的个数，默认为5个
    def ployinterp_column(s, n, k=5):
        y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
        y = y[y.notnull()]  # 剔除异常值
        return lagrange(y.index, list(y))(n)

    for i in data.columns:
        for j in range(len(data)):
            if (data[i].isnull())[j]:
                data[i][j] = ployinterp_column(data[i], j)

    data.to_excel(outputfile, header=None, index=False)

def cm_plot(y,yp):
    cm = confusion_matrix(y,yp)

    plt.matshow(cm,cmap=plt.cm.Greens)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(
                cm[x,y],
                xy=(x,y),
                horizontalalignment='center',
                verticalalignment='center'
            )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

def lm_model():
    datafile = 'data/model.xls'
    data = pd.read_excel(datafile)
    data = data.as_matrix()
    shuffle(data)

    p = 0.8
    train = data[:int(len(data) * p), :]
    test = data[int(len(data) * p):, :]

    # 构建LM神经网络模型
    netfile = 'tmp/net.model'

    net = Sequential()  # 建立神经网络
    #    net.add(Dense(input_dim = 3, units = 10))
    # 添加输入层（3节点）到隐藏层（10节点）的连接
    net.add(Dense(10, input_shape=(3,)))
    net.add(Activation('relu'))  # 隐藏层使用relu激活函数
    #    net.add(Dense(input_dim = 10, units = 1))
    # 添加隐藏层（10节点）到输出层（1节点）的连接
    net.add(Dense(1, input_shape=(10,)))
    net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
    net.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        sample_weight_mode="binary")  # 编译模型，使用adam方法求解

    net.fit(train[:, :3], train[:, 3], epochs=100, batch_size=1)
    net.save_weights(netfile)

    predict_result = net.predict_classes(train[:, :3]).reshape(
        len(train))  # 预测结果变形
    '''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''

    cm_plot(train[:, 3], predict_result).show()

    predict_result = net.predict(test[:, :3]).reshape(len(test))
    fpr, tpr, thresholds = roc_curve(test[:, 3], predict_result, pos_label=1)
    plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend(loc=4)
    plt.show()
    print(thresholds)



if __name__ == '__main__':
    Lagrange_interpolation()
    lm_model()
