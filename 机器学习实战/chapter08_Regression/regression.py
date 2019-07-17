# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/14 20:25'

import numpy as np
import time


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split("\t"))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = denom.I * xMat.T * yMat
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 20
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


from urllib import request
from bs4 import BeautifulSoup
import json


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    time.sleep(10)
    myAPIstr = "AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY"
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict["items"])):
        try:
            currItem = retDict["items"][i]
            if currItem["product"]["inventories"] == "new":
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem["product"]["inventories"]
            for item in listOfInv:
                sellingPrice = item["price"]
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal,30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        np.random.shuffle(indexList)
        for j in range(m):
            if(j < m * 0.9):
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
            wMat = ridgeTest(trainX,trainY)
            for k in range(30):
                matTestX = np.mat(testX)
                matTrainX = np.mat(trainX)
                meanTrain = np.mean(matTrainX,0)
                varTrain = np.var(matTrainX,0)
                matTestX = (matTestX-meanTrain)/varTrain
                yEst = matTestX * np.mat(wMat[i,:]).T + np.mean(trainY)
                errorMat[i,k] = rssError(yEst.T.A,np.array(testY))
    meanErrors = np.mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors==minMean)]
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0)
    varX = np.var(xMat,0)
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*np.sum(np.multiply(meanX,unReg)) + np.mean(yMat))

if __name__ == '__main__':
    pass
    # xArr,yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr,yArr)
    # xMat = np.mat(xArr)
    # yMat = np.mat(yArr)
    # yHat = xMat * ws
    # import matplotlib.pyplot as plt
    # fig= plt.figure()
    # ax = fig.add_subplot(111)
    # print(yMat)
    # ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0])
    # yHat = xMat*ws
    # yHatMax = yHat.argmax()
    # yHatMin = yHat.argmin()
    # print([xMat[yHatMin,1],yHat[yHatMin].flatten().A[0][0]])
    # print([xMat[yHatMax,1],yHat[yHatMax].flatten().A[0][0]])
    # ax.plot([xMat[yHatMin,1],xMat[yHatMax,1]],[yHat[yHatMin].flatten().A[0][0],yHat[yHatMax].flatten().A[0][0]],'r-')
    # np.corrcoef(yHat.T,yMat)
    # plt.show()
    # abX, abY = loadDataSet("abalone.txt")
    # stageWise(abX, abY, 0.01, 200)
    # ridgeWeights = ridgeTest(abX,abY)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()
    # lgX = []
    # lgY = []
    # setDataCollect(lgX,lgY)
