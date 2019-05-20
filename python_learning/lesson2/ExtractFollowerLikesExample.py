# -*- coding: utf-8 -*-

# inFile: findBothFacebookTwitterAll
# twitterID   #followr FacebookID     #likes                         
# 40053156	28	rwilderjr	0	
# 30693817	488	cookiemodel	0	
#inFile2: TwitterProfileData

#     userID         Name          screenName          Location      Description                    Url              Protected Follower Friend  
# |::|21579600|::|Sarah Fleenor|::|SarahFleenor|::|Indianapolis|::|25.Has.......IUPUI Alum.|::|http://t.co/F0RCZAemX4|::|False|::|353|::|382|::|
#       0             1               2                  3               4                           5                     6       7      8
#    CreatedAt       Favourite  UtcOffset         TimeZone              Status   Lang List   lastTweetTime           lastTweet
# 2009-02-22 17:18:31|::|1241|::|-14400|::|Eastern Time (US & Canada)|::|18689|::|en|::|8|::|2016-04-14 22:36:08|::|{'contributors': .... None}|;;|
#       9                 10       11               12                    13      14    15        16                    17

import re
import datetime
import time
import sys
import os
import traceback 

def readBothFile(inFile,outFile,twitterFollower):
    fout = open(outFile,"w")
    fin = open(inFile,encoding='utf-8') # findBothFacebookTwitterAll
    for current in fin:
        current = current.replace('\n','')
        curL = current.split('\t')
        twitter = curL[0]
        #follower = curL[1]
        facebook = curL[2]
        likes = curL[3]
        if (twitter in twitterFollower):
            follower = twitterFollower[twitter]
            fout.write(twitter + '\t' + follower + '\t' + facebook + '\t' + likes +  '\n')

def extractFollower(inFile2,twitterFollower):
    columnMark =  '|::|'
    rowMark = '|;;|\n'
    count = 0
    fin = open(inFile2,encoding='utf-8') # TwitterUserProfile
    for current in fin:
        #print (current[0:4])
        if not (current[0:4] == columnMark and current[-5:] == rowMark): 
            continue 
        count += 1 # allCount
        if(count % 100000 == 0):
            print ('twitterFollower:' + str(count))
        data = current[4:-5]
        data = data.replace('\n','')
        curL = data.split(columnMark)
        userId = curL[0]
        screenName = curL[2]
        follower = curL[7]
        if (not userId in twitterFollower):
            twitterFollower[userId] = follower
        if (not screenName in twitterFollower):
            twitterFollower[screenName] = follower 
    print ('twitterFollower is done:' + str(len(twitterFollower)))

def main(argv):
    inFile = argv[1]  # findBothFacebookTwitterAll
    inFile2 = argv[2]  # TwitterUserProfile
    outFile = argv[3]  # findBothFacebookTwitterAllUpdate

    #twitterFacebook = {} # twitterFacebook[twitterAccount + facebookAccount] = [#followr, #likes, Url]
    twitterFollower = {} # twitterFollower[user] = #followr
    extractFollower(inFile2,twitterFollower)
    
    
    readBothFile(inFile,outFile,twitterFollower)


if __name__ == "__main__":
    #ִmain fuction
    main(sys.argv)

