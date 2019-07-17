# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/15 16:38'

import numpy as np


class treeNode:
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float, curLine)
        dataMat.append(list(fltLine))
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestValue = 0
    bestIndex = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree["spInd"] = feat
    retTree["spVal"] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree["left"] = createTree(lSet, leafType, errType, ops)
    retTree["right"] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == "dict")


def getMean(tree):
    if isTree(tree["right"]): tree["right"] = getMean(tree["right"])
    if isTree(tree["left"]): tree["left"] = getMean(tree["left"])
    return (tree["left"] + tree["right"]) / 2


def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)
    if (isTree(tree["right"])) or (isTree(tree["left"])):
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])
    if isTree(tree["left"]): tree["left"] = prune(tree["left"], lSet)
    if isTree(tree["right"]): tree["right"] = prune(tree["right"], rSet)
    if not isTree(tree["left"]) and not isTree(tree["right"]):
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree["left"], 2)) + np.sum(
            np.power(rSet[:, -1] - tree["right"], 2))
        treeMean = (tree["left"] + tree["right"]) / 2
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        raise NameError("This matrix is singular,cannot do inverse,\n try increase the second value of ops")
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y- yHat,2))

def regTreeEval(model,inData):
    return model

def modelTreeEval(model,inData):
    n = np.shape(inData)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inData
    return X * model

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):return modelEval(tree,inData)
    if inData[tree["spInd"]] > tree["spVal"]:
        if isTree(tree["left"]):
            return treeForeCast(tree["left"],inData,modelEval)
        else:
            return modelEval(tree["left"],inData)
    else:
        if isTree(tree["right"]):
            return treeForeCast(tree["right"],inData,modelEval)
        else:
            return modelEval(tree["right"],inData)
def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat
if __name__ == '__main__':
    pass
    # dataSet = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # print(dataSet[:,1])
    # print(dataSet[:,1] > 3)
    # print(np.nonzero(dataSet[:,1] > 3))
    # testMat = np.mat(np.eye(4))
    # print(testMat)
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print(mat0)
    # myDat1 = loadDataSet("ex0.txt")
    # myMat1 = np.mat(myDat1)
    # tree = createTree(myMat1)
    # print(tree)
    import tkinter as tk
    root = tk.Tk()
    myLabel = tk.Label(root,text="Hello World")
    myLabel.grid()
    root.mainloop()
