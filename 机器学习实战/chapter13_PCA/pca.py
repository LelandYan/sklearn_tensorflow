# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/18 8:08'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim="\t"):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, 0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet("secom.data"," ")
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i])
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat


if __name__ == '__main__':
    # dataMat = loadDataSet("testSet.txt")
    # lowDMat,reconMat = pca(dataMat,1)
    # # print(np.shape(lowDMat))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker="^",s=90)
    # ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker="o",s=50,c="red")
    # plt.show()
    dataMat = replaceNanWithMean()
    meanVals =np.mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved,rowvar=False)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    print(eigVals)
