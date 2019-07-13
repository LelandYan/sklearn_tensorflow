# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/13 15:12'
import cmath
import random
import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱 0表示正常言论
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWordsVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"the word: {word} is not in my Vocabulary")
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)  # 防止贝叶斯公式中其中一个概率值为0 本来使用zeros
    p1Num = np.ones(numWords)  # 防止贝叶斯公式中其中一个概率值为0 本来使用zeros
    p0Denom, p1Denom = 2.0, 2.0  # 防止贝叶斯公式中其中一个概率值为0  本来使用0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum((trainMatrix[i]))
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + cmath.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + cmath.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWordsVec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ["love", "my", "delmation"]
    thisDoc = np.array(setOfWordsVec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ["stupid", "garbage"]
    thisDoc = np.array(setOfWordsVec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0v, p1v, pAb))


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    for i in range(1,26):
        wordList = textParse(open(f"email/spam/{i}.txt").read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open(f"email/ham/{i}.txt").read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWordsVec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errroCount = 0
    for docIndex in testSet:
        wordVector = setOfWordsVec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:errroCount+=1
    print("the error rate is: ",float(errroCount)/len(testSet))

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # print(setOfWordsVec(myVocabList,listOPosts[1]))
    # print([1,2,3] +[0])
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWordsVec(myVocabList, postinDoc))
    # p0v, p1v, pAb = trainNB0(trainMat, listClasses)
    # print(pAb)
    # print(p0v)
    # testingNB()
    # regEx = re.compile("\\W*")
    spamTest()