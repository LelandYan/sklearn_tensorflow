# -*- coding: utf-8 -*-

#extract feed features from FormatTwitterTweets

# input1: IdentifySameUserTendUrl
# twitterID #followr #twitterUrl #sameLongUrl FacebookID #likes #facebookUrl     howMatchUserID        twitterName       facebookName   UtcOffset 
# 109118967	3904208	41	    5	113525172018777	2464791	12	FacebookFromTwitterProfiles	jaimecamil	jaimecamil      -14400
# 62477022	244525	15	    7	192465074360	1724259	14	FacebookFromTwitterProfiles	andaadam	AndaAdamOfficial -18000
#   0           1       2           3           4         5     6                    7                   8                   9             10

# input2: TwitterTweetsFormat:
#    userID       TweetID            #RT #favorite #comment    date         isRT isRep      expandURL    #mentions #hashtage   text
#|::|14985126|::|714569507218456577|::|1|::|2|::|0|::|2016-03-28 21:47:01|::|0|::|0|::|http://bit.ly/1LammK9|::|0|::|0|::|What to https://t.co/WdNuNLJ0Mu|;;|
#|::|14985126|::|714520937341784064|::|2|::|3|::|0|::|2016-03-28 18:34:01|::|0|::|0|::|http://bit.ly/1o0Yy0R|::|0|::|0|::|When it https://t.co/ad5Go6wy7q https://t.co/N06Lnne6yE|;;|
#    0                  1              2    3    4               5           6    7            8                9    10        11 

# output: TwitterTweetFeature
# userID	tweetID	        #mentions	#hashtage	lengthContent	    week    created_time	Density1	Density2
# 37097199	656198344528674816	0	0	        95	             1       220	        1	        2
# 37097199	656181145583751168	0	0	        23	             2        218	        0	        0
#   0            1                      2       3               4                    5        6                 7               8

import sys
import os
import traceback
import re
import datetime
import time

def readSelectUsers(inFile,TwitterSelectUser):
    fin = open(inFile) 
    count =0
    selectCount = 0
    for current in fin:
        count += 1
        data = current.replace('\n','')
        curL = data.split('\t')
        twitterID = curL[0]
        UtcOffset = int(curL[10])/3600
        TwitterSelectUser[twitterID] = UtcOffset
def readFile(inFile2,twitterData,TwitterSelectUser):
    fin = open(inFile2)
    columnMark = '|::|'
    rowMark = '|;;|\n'  
    count = 0
    
    for current in fin:
        count += 1
        if not(current[0:4]==columnMark and current[-5:]==rowMark):
            print ("error--current[0:4]==columnMark and current[-5:]==rowMark-"  )
        if (count % 40000 ==0):
            print (count)
        data = current[4:-5]        
        dataArray = data.split('|::|')        
        
        userID = dataArray[0]
        if (not userID in TwitterSelectUser):
            continue
        #created_time = datetime.datetime.strptime(dataArray[5],"%Y-%m-%dT%H:%M:%S+0000")
        #created_time = datetime.datetime.strptime(dataArray[5],"%Y-%m-%d %H:%M:%S") 
        originalTime = datetime.datetime.strptime(dataArray[5],"%Y-%m-%d %H:%M:%S")
        adjustTime = originalTime + datetime.timedelta(hours = TwitterSelectUser[userID])  # ---------------------timezone----adjuct----------------
        adjustTimeStr = adjustTime.strftime("%Y-%m-%d %H:%M:%S")
        created_time = datetime.datetime.strptime(adjustTimeStr,"%Y-%m-%d %H:%M:%S")        
        
        
        twitterData.setdefault(userID,[]).append(created_time)
    #twitterData[userID].sort()
    fin.close()
def twitterDataSort(twitterData):                             #排序函数(逆序排序，原因，对比的时候更省时间) 
    count = 0
    for userID in twitterData:
        count += 1
        if (count % 400 ==0):
            print (count)
        twitterData[userID].sort(reverse=True)
        
def writeFile(inFile2,twitterData, TwitterSelectUser, outFile):
    fin = open(inFile2)
    fout = open(outFile,'w')
    
    columnMark = '|::|'
    rowMark = '|;;|\n'    
    count = 0
    
    for current in fin:
        count += 1
        if not(current[0:4]==columnMark and current[-5:]==rowMark):
            print ("error--current[0:4]==columnMark and current[-5:]==rowMark-")
        if (count % 10000 ==0):
            print (count)
        data = current[4:-5]        
        dataArray = data.split('|::|')
        
        userID = dataArray[0]
        if (not userID in TwitterSelectUser):
            continue        
        feedID = dataArray[1]
        mentions = dataArray[9]                 
        hashtage = dataArray[10]
        lengthContent = 0
        if(dataArray[11] != 'null'):
            lengthContent = len(dataArray[11])  
            

        #created_time = datetime.datetime.strptime(dataArray[5],"%Y-%m-%dT%H:%M:%S+0000")
        #created_times = datetime.datetime.strptime(dataArray[5],"%Y-%m-%d %H:%M:%S")
        originalTime = datetime.datetime.strptime(dataArray[5],"%Y-%m-%d %H:%M:%S")
        adjustTime = originalTime + datetime.timedelta(hours = TwitterSelectUser[userID])  # ---------------------timezone----adjuct----------------
        adjustTimeStr = adjustTime.strftime("%Y-%m-%d %H:%M:%S")
        created_time = datetime.datetime.strptime(adjustTimeStr,"%Y-%m-%d %H:%M:%S")

        (whichWeek,fmaTime) = formatTime(created_time)
        (countOneHour,conutTwoHour) = extractCount(userID, created_time,twitterData)

        fout.write(str(userID) + '\t' + str(feedID) + '\t' + str(mentions)+ '\t' + str(hashtage)  + '\t' +
                   str(lengthContent) + '\t' + str(whichWeek) + '\t' + str(fmaTime) + '\t' + str(countOneHour) + '\t' + str(conutTwoHour) + '\n')        
    fout.close()
    
#下面now.weekday()
# 星期一 |星期二 星期三 星期四 星期五 星期六 星期七
#   0    |   1     2      3      4      5      6
#下面是now.isoweekday()
# 星期一 |星期二 星期三 星期四 星期五 星期六 星期七
#   1    |   2     3      4      5      6      7
def formatTime(created_time): # 2016-03-28 21:47:01 output: 121
    day = created_time.isoweekday()
    day2 = 1
    if(day==6 or day ==7):
        day2 = 0
    timeTwo = str(created_time)[11:13]
    return (day2,timeTwo)

def extractCount(userID,created_time,twitterData):
    countOneHour = 0
    conutTwoHour = 0  
    
    #for checkTime in twitterData[userID]:
    indexcount = twitterData[userID].index(created_time)           #获取发布时间在列表中的位置
    if (indexcount == len(twitterData[userID])):
        return (countOneHour,conutTwoHour)
    else:
        timeOne = created_time-datetime.timedelta(hours=1)           #获取发布前一个小时时间
        timeTwo = created_time-datetime.timedelta(hours=2)           #获取发布前两个小时时间 
        for checkTime in twitterData[userID][indexcount+1:]:
            if(checkTime < timeTwo):
                return (countOneHour,conutTwoHour)
            else:
                if(checkTime >= timeOne):
                    countOneHour += 1
                #else:
                #    conutTwoHour += 1
                conutTwoHour += 1
    return (countOneHour,conutTwoHour)
    
def printTime(beginTime):
    endTime = datetime.datetime.now() #calculate time
    print ("------------consumed-----------time-----------begin-----------")
    print ("consumed time:" + str(endTime - beginTime) )
    print ("------------consumed-----------time-----------end-------------")     
    
def main(argv):
    inFile = argv[1]  # IdentifySameUserTendUrl
    inFile2 = argv[2] # FormatTwitterTweets
    outFile = argv[3] # TwitterTweetFeature

    beginTime = datetime.datetime.now()
    print ('now ,beginTime is:' + str(beginTime) )
    
    TwitterSelectUser = {}  # TwitterSelectUser[userID]  = UtcOffset
    readSelectUsers(inFile,TwitterSelectUser)

    twitterData = {}    # twitterData[userID] = [time1, time2, time3......]
    print ('begin to read file')
    readFile(inFile2, twitterData,TwitterSelectUser)
    print ('begin to sort dict')
    twitterDataSort(twitterData)
    print ('begin to write file')
    writeFile(inFile2, twitterData, TwitterSelectUser,outFile)                  
                                     
                                     
    printTime(beginTime)
    print ('\a')
    print ('finish')
                    
if __name__=="__main__":
    main(sys.argv)