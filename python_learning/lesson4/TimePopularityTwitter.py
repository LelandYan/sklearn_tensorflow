import sys
import datetime


def readSelectUsers(inFile, TwitterSelectUser):
    fin = open(inFile)
    count = 0
    for current in fin:
        count += 1
        data = current.replace("\n", "")
        curL = data.split("\t")
        twitterID = curL[0]
        UtcOffset = int(curL[10]) / 3600
        TwitterSelectUser[twitterID] = UtcOffset


def readFile(inFile2, TwitterData, TwitterSelectUser):
    fin = open(inFile2)
    columnMark = '|::|'
    rowMark = '|;;|\n'
    count = 0

    for current in fin:
        count += 1
        if not (current[0:4] == columnMark and current[-5:] == rowMark):
            print("error--current[0:4]==columnMark and current[-5:]==rowMark-")
        if (count % 40000 == 0):
            print(count)
        data = current[4:-5]
        dataArray = data.split('|::|')

        userID = dataArray[0]
        if (not userID in TwitterSelectUser):
            continue
        originalTime = datetime.datetime.strptime(dataArray[5], "%Y-%m-%d %H:%M:%S")
        adjustTime = originalTime + datetime.timedelta(
            hours=TwitterSelectUser[userID])  # ---------------------timezone----adjuct----------------
        adjustTimeStr = adjustTime.strftime("%Y-%m-%d %H:%M:%S")
        created_time = datetime.datetime.strptime(adjustTimeStr, "%Y-%m-%d %H:%M:%S")

        TwitterData.setdefault(userID, []).append(created_time)
    fin.close()


def twitterDataSort(twitterData):
    count = 0
    for userID in twitterData:
        count += 1
        if (count % 400 == 0):
            print(count)
        twitterData[userID].sort(reverse=True)


def writeFile(inFile2, TwitterSelectUser, outFile):
    fin = open(inFile2)
    columnMark = '|::|'
    rowMark = '|;;|\n'
    count = 0
    res = {i:[0 for _ in range(3)] for i in range(24)}
    for current in fin:
        count += 1
        if not (current[0:4] == columnMark and current[-5:] == rowMark):
            print("error--current[0:4]==columnMark and current[-5:]==rowMark-")
        if (count % 10000 == 0):
            print(count)
        data = current[4:-5]
        dataArray = data.split('|::|')

        userID = dataArray[0]
        if (not userID in TwitterSelectUser):
            continue

        originalTime = datetime.datetime.strptime(dataArray[5], "%Y-%m-%d %H:%M:%S")
        adjustTime = originalTime + datetime.timedelta(
            hours=TwitterSelectUser[userID])  # ---------------------timezone----adjuct----------------
        adjustTimeStr = adjustTime.strftime("%Y-%m-%d %H:%M:%S")
        created_time = datetime.datetime.strptime(adjustTimeStr, "%Y-%m-%d %H:%M:%S")
        
        (whichWeek, fmaTime) = formatTime(created_time)
        res[fmaTime][whichWeek] += 1
    with open(outFile, 'w') as f:
        for key,lists in res.items():
            f.write(str(key+1))
            for click in lists:
                    f.write("\t"+str(click))
            f.write("\n")

def formatTime(created_time):  # 2016-03-28 21:47:01 output: 121
    day = created_time.isoweekday()
    if (day >= 1 and day <= 5):
        day = 0
    elif(day == 6):
        day = 1
    elif (day == 7):
        day = 2
    timeTwo = str(created_time)[11:13]
    if str(created_time)[11:13] == "00" or str(created_time)[11:13] == "0":
        timeTwo = "24"
    return (day, int(timeTwo)-1)

def printTime(beginTime):
    endTime = datetime.datetime.now() #calculate time
    print ("------------consumed-----------time-----------begin-----------")
    print ("consumed time:" + str(endTime - beginTime) )
    print ("------------consumed-----------time-----------end-------------")
def main(argv):
    inFile = argv[1]  # IdentifySameUserTendUrl
    inFile2 = argv[2] # FormatTwitterTweets
    outFile = argv[3] # TimePopularityTwitter
    # inFile = "IdentifySameUserTendUrl"
    # inFile2 = "FormatTwitterTweets"
    # outFile = "TimePopularityTwitter"

    beginTime = datetime.datetime.now()
    print("now,beginTime is:", str(beginTime))

    TwitterSelectUser = {}
    readSelectUsers(inFile, TwitterSelectUser)

    twitterData = {}
    print("begin to read file")
    readFile(inFile2, twitterData, TwitterSelectUser)
    print("begin to sort dict")
    twitterDataSort(twitterData)
    print("begin to write file")
    writeFile(inFile2, TwitterSelectUser, outFile)
    printTime(beginTime)
    print('\a')
    print('finish')

if __name__ == '__main__':
    main(sys.argv)
    # main(1)
