# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/23 15:17'
import sys
import datetime
import os


def print_time(time_begin):
    time_end = datetime.datetime.now()
    time_cost = time_end - time_begin
    print("cost " + str(time_cost.seconds) + "s")


def getVideoDict(inFile, videoDict):
    print("begin to getVideoDict")
    fin = open(inFile, 'r')
    for current in fin:
        vid = current.replace('\n', '')
        videoDict[vid] = {}
    # print ("*********end*********")
    print("Number of video:" + str(len(videoDict)))


def readViewFile(pathFile, videoDict, fileDate):
    # print("begin to read file" + pathFile)
    fin = open(pathFile, 'r')
    for current in fin:
        data = current.split('\t')
        video = data[0]
        viewCount = data[1]
        if (video in videoDict):
            videoDict[video][fileDate] = viewCount
    # print ("*********end*********")


def main(argv):
    time_begin = datetime.datetime.now()
    # inFile = argv[1]  # selectedVideoID
    # inFile2 = argv[2]  # dataTest
    # outFile = argv[3]  # VideoDateView
    inFile = "selectedVideoID"
    inFile2 = "dataTest"
    outFile = "VideoViewDateRow"
    # 获取id的列表
    videoDict = {}  # videoDict[videoId] = {} # videoDict[videoId][date] = viewCount
    getVideoDict(inFile, videoDict)

    # 读取每一天的数据，把观看数保存在dict

    count = 0
    for root, dirs, files in os.walk(inFile2):
        for filename in files:
            count += 1
            yearPos = filename.find('_2017')
            fileDate = filename[yearPos + 1:-6]
            # print fileDate
            # print filename
            pathFile = inFile2 + '/' + filename
            print(str(count) + '-----' + pathFile)
            readViewFile(pathFile, videoDict, fileDate)  # videoDict[videoId][date] = viewCount

    fout = open(outFile, 'w')
    for (key, value) in videoDict.items():  # key is videoId
        content = ''
        # dateView = value.items()
        # dateView.sort()
        dateView = sorted(value.items())
        for (u, v) in dateView:  # u is date, v is viewCount
            fout.write(key + "\t" + u + "\t" + v + "\n")
            #content += v + '\t'  # ---------------------diffent 1 from VideoViewDateRow.py---------------
        #fout.write(key + '\t' + content + '\n')  # ---------------------diffent 2 from VideoViewDateRow.py---------------

    print_time(time_begin)
    print("finish")


if __name__ == '__main__':
    # main(sys.argv)
    main(1)