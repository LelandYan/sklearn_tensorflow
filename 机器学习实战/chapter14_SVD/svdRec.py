# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/18 8:44'

import numpy as np


def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


def euclidSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print(f"the {item} and {j} similarity is :{similarity}")
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0: return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def svdEst(dataMat,user,simMeans,item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT= np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4)*Sigma[:4])
    xformedItems = dataMat.T * U[:,4] * Sig4.I
    for j in range(n):
        userRating = simMeans(xformedItems[item,:].T,xformedItems[j,:].T)
        if userRating == 0 or j == item:continue
        similarity = simMeans(xformedItems[item,:].T,xformedItems[j,:].T)
        print(f"the {item} and {j} similarity is :{similarity}")
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:return 0
    else:return ratSimTotal / simTotal



if __name__ == '__main__':
    myMat = np.mat(loadExData())
    myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
    myMat[3,3] = 2
    print(myMat)
    print(recommend(myMat,2))