# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/6/5 20:53'

sen = {}
with open('sentimentLabel.txt','r') as f:
    for line in f:
        line = line.replace("\n","")+""
        sen[line.split("\t")[1]] = line.split("\t")[0]


with open("test.txt","r") as f:
    cnt = 0
    t = 0
    for line in f:
        line = line.replace("\n", "")
        posi = 0
        nega = 0
        line = line.replace("\t", " ")
        for word in line.split(" ")[1:]:
            p = 0
            n = 0
            for key,value in sen.items():
                if word in key and value == "Positive":
                    p += 1
                if word in key and value == "Negative":
                    n += 1
            posi += p
            nega += n
        print(posi,nega)
        if posi >= nega:
            if line.split(" ")[0] == "Positive":
                t+=1
            print("Pos"+"::"+line)
        if nega > posi:
            if line.split(" ")[0] == "Negative":
                t += 1
            print("Neg"+"::"+line)
        cnt += 1
    print(t / cnt)
