# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/13 8:26'

import os
import operator
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistanceIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistanceIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet - np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(f"the classifier came back with {classifierResult},the real answer is:{datingLabels[i]}")
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print(f"the total error rate is :{errorCount/numTestVecs}")


def img2Vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[i])

    return returnVect

def handwritingClassTest():
    hwLabels = []
    # 遍历所有的训练集文件夹下的文件
    trainingFileList = os.listdir("trainingDigits")
    # 获取训练集文件夹下的文件的数量
    m = len(trainingFileList)
    # 将所有的训练集都放入一个矩阵中
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr= fileNameStr.split('.')[0] # 0.0
        classNumStr = int(fileStr.split('_')[0]) # 0
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2Vector(f'trainingDigits/{fileNameStr}')
    testFileList = os.listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector(f'testDigits/{fileNameStr}')
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print(f'the classifier came back with:{classifierResult},the real answer is:{classNumStr}')
        if(classifierResult != classNumStr):errorCount+=1.0
    print(f"the total the number of errors is {errorCount}")
    print(f"the total error rate is:{errorCount/mTest}")
if __name__ == '__main__':
    # group, labels = createDataSet()
    # print(classify0([0, 0], group, labels, 3))
    # datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
    # plt.show()
    # normMat,ranges,minVals = autoNorm(datingDataMat)
    # datingClassTest()
    testVect = img2Vector("testDigits/0_13.txt")
    # print(testVect[0,0:31])
    handwritingClassTest()