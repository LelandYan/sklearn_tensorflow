# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/20 20:32'

import sys


def readBothFile(inFile, outFile, twitterFollower, twitterLike):
    fout = open(outFile, "w")
    fin = open(inFile, encoding='utf-8')  # findBothFacebookTwitterAll
    for current in fin:
        current = current.replace('\n', '')
        curL = current.split('\t')
        twitter = curL[0]
        # follower = curL[1]
        facebook = curL[2]
        likes = curL[3]
        if (twitter in twitterFollower) and (facebook in twitterLike):
            follower = twitterFollower[twitter]
            likes = twitterLike[facebook]
            fout.write(twitter + '\t' + follower + '\t' + facebook + '\t' + likes + '\n')


def extractFollower(inFile2, twitterFollower):
    columnMark = '|::|'
    rowMark = '|;;|\n'
    count = 0
    fin = open(inFile2, encoding='utf-8')  # TwitterUserProfile
    for current in fin:
        if not (current[0:4] == columnMark and current[-5:] == rowMark):
            continue
        count += 1  # allCount
        if (count % 100000 == 0):
            print('twitterFollower:' + str(count))
        data = current[4:-5]
        data = data.replace('\n', '')
        curL = data.split(columnMark)
        userId = curL[0]
        screenName = curL[2]
        follower = curL[7]
        if (not userId in twitterFollower):
            twitterFollower[userId] = follower
    print('twitterFollower is done:' + str(len(twitterFollower)))


def extractLikes(inFile3, twitterLikes):
    columnMark = '|::|'
    rowMark = '|;;|\n'
    count = 0
    fin = open(inFile3, encoding='utf-8')  # TwitterUserProfile
    for current in fin:
        if not (current[0:4] == columnMark and current[-5:] == rowMark):
            continue
        count += 1  # allCount
        if (count % 100000 == 0):
            print('twitterLike:' + str(count))
        data = current[4:-5]
        data = data.replace('\n', '')
        curL = data.split(columnMark)
        facebookID = curL[1]
        like = curL[2]
        if (not facebookID in twitterLikes):
            twitterLikes[facebookID] = like
    print('twitterLike is done:' + str(len(twitterLikes)))


def main(argv):
    inFile = argv[1]  # findBothFacebookTwitterAll
    inFile2 = argv[2]  # TwitterUserProfile
    inFile3 = argv[3]  # FacebookUserProfile
    outFile = argv[4]  # ExtractFollowerLikesExample2

    # twitterFacebook = {} # twitterFacebook[twitterAccount + facebookAccount] = [#followr, #likes, Url]
    twitterFollower = {}  # twitterFollower[user] = #followr
    twitterLike = {}
    extractFollower(inFile2, twitterFollower)
    extractLikes(inFile3, twitterLike)
    readBothFile(inFile, outFile, twitterFollower, twitterLike)


if __name__ == "__main__":
    # ִmain fuction
    main(sys.argv)
