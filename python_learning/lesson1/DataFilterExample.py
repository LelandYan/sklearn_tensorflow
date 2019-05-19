# -*- coding: utf-8 -*-
# select users  whose timeZone is not "-1"

# input: UserInfoOriginal    
# twitterID #followr #twitterUrl #sameLongUrl FacebookID #likes #facebookUrl     howMatchUserID        twitterName       facebookName   timeZone 
# 109118967	3904208	41	    5	113525172018777	2464791	12	FacebookFromTwitterProfiles	jaimecamil	jaimecamil      -14400
# 62477022	244525	15	    7	192465074360	1724259	14	FacebookFromTwitterProfiles	andaadam	AndaAdamOfficial -18000
#   0           1       2           3           4         5     6                    7                   8                   9             10

# output: UserInfoFilter1    
# twitterID #followr #twitterUrl #sameLongUrl FacebookID #likes #facebookUrl     howMatchUserID        twitterName       facebookName   timeZone 
# 109118967	3904208	41	    5	113525172018777	2464791	12	FacebookFromTwitterProfiles	jaimecamil	jaimecamil      -14400
# 62477022	244525	15	    7	192465074360	1724259	14	FacebookFromTwitterProfiles	andaadam	AndaAdamOfficial -18000
#   0           1       2           3           4         5     6                    7                   8                   9             10


import datetime
import time
import sys
import os



def printTime(beginTime):
    endTime = datetime.datetime.now() #calculate time
    print ("------------consumed-----------time-----------begin-----------")
    print ("consumed time:" + str(endTime - beginTime) )
    print ("------------consumed-----------time-----------end-------------")


def readUserPair(inFile,outFile):
    fin = open(inFile)   # UserInfoOriginal
    # twitterID #followr #twitterUrl #sameLongUrl FacebookID #likes #facebookUrl     howMatchUserID        twitterName       facebookName   timeZone 
    # 109118967	3904208	41	    5	113525172018777	2464791	12	FacebookFromTwitterProfiles	jaimecamil	jaimecamil      -14400
    #   0           1       2           3           4         5     6                    7                   8                   9             10
    
    fout = open(outFile,'w')
    count =0
    for current in fin:
        count += 1
        if(count % 1000 == 0):
            print (count)
        data = current.replace('\n','')
        curL = data.split('\t')
        #twitterID = curL[0]
        #twFollower = int(curL[1])
        #facebookID = curL[4]
        #fbLikes = int(curL[5])
        timezone = curL[10]

        if(timezone == "-1" ):
            continue
        fout.write(current)        
                  
def main(argv):
    inFile = argv[1]   # UserInfoOriginal 
    outFile = argv[2]  # UserInfoFilter1
    beginTime = datetime.datetime.now() 

    
    readUserPair(inFile,outFile)
    
  
    
    printTime(beginTime)       
    print ("\a")
    print ('finish' )

    
if __name__ == "__main__":
    #ִmain fuction
    main(sys.argv)
