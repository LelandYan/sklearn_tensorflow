# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/14 9:12'
import numpy as np


def loadDataSet(fileName):
    """
    导入数据
    :param fileName:文件名
    :return:
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    在某个区间范围捏随机选择一个整数，只要函数值不等于输入值i
    :param i:是第一个alpha的下标
    :param m:是所有alpha的数目
    :return:
    """
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# SMO优化算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    使用SMO算法求解SVM参数
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 最大的循环次数
    :return:
    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print(f"iter: {iter} i:{i}, paris changed {alphaPairsChanged}")
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print(f"iteration number: {iter}")
    return b, alphas


# class optStruct:
#     def __init__(self, dataMatMin, claasLabels, C, toler):
#         self.X = dataMatMin
#         self.labelMat = claasLabels
#         self.C = C
#         self.tol = toler
#         self.m = np.shape(dataMatMin)[0]
#         self.alphas = np.mat(np.zeros((self.m, 1)))
#         self.b = 0
#         self.eCache = np.mat(np.zeros((self.m, 2)))
#
#
# def calcEk(oS, k):
#     fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
#     Ek = fXk - float(oS.labelMat[k])
#     return Ek
#
# def selectJ(i,oS,Ei):
#     maxK = -1
#     maxDeltaE = 0
#     Ej = 0
#     oS.eCache[i] = [1,Ei]
#     validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
#     if(len(validEcacheList)) > 1:
#         for k in validEcacheList:
#             if k == i:continue
#             Ek = calcEk(oS,k)
#             deltaE = abs(Ei - Ek)
#             if(deltaE > maxDeltaE):
#                 maxK = k
#                 maxDeltaE = deltaE
#                 Ej = Ek
#         return maxK,Ej
#     else:
#         j = selectJrand(i,oS.m)
#         Ej = calcEk(oS.j)
#     return j,Ej
#
# def updateEk(oS,k):
#     Ek = calcEk(oS,k)
#     oS.eCache[k] = [1,Ek]
#
# def innterL(i,oS):
#     Ei = calcEk(oS,i)
#     if(((oS.labelMat[i]*Ei) < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] > oS.tol) and (oS.alphas[i] > 0)):
#         j,Ej = selectJ(i,oS,Ei)
#         alphaIold = oS.alphas[i].copy()
#         alphaJold = oS.alphas[j].copy()
#         if (oS.labelMat[i] != oS.labelMat[j]):
#             L = max(0,oS.alphas[j] - oS.alphas[i])
#             H = min(oS.C,oS.C+oS.alphas[j] - oS.alphas[i])
#         else:
#             L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
#             H = min(oS.C, oS.alphas[j] + oS.alphas[i])
#             pass
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == "lin":
        K = X * A.T
    elif kTup[0] == "rbf":
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError("Houston We Have a Problem That Kernel is not recognized")
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet("testSet.txt")
    # print(labelArr)
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
