# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/20 19:52'

import datetime
import sys


def printTime(beginTime):
    endTime = datetime.datetime.now()
    print("------------consumed-----------time-----------begin-----------")
    print("consumed time:" + str(endTime - beginTime))
    print("------------consumed-----------time-----------end-------------")


def readUserPair(inFile, outFile):
    fin = open(inFile)

    fout = open(outFile,'w')
    count = 0
    for current in fin:
        count += 1
        if (count % 1000 == 0):
            print(count)
        data = current.replace('\n', '')
        curl = data.split("\t")
        follower = int(curl[1])
        likes = int(curl[5])
        if follower < likes:
            follower, likes = likes, follower
        if (follower // likes) > 10:
            continue
        fout.write(current)


def main(argv):
    inFile = argv[1]
    outFile = argv[2]
    # inFile = "UserInfoOriginal"
    # outFile = "UserInfoFilter2"
    beginTime = datetime.datetime.now()

    readUserPair(inFile, outFile)

    printTime(beginTime)
    print("\a")
    print("finish")


if __name__ == '__main__':
    main(sys.argv)
    # main(1)