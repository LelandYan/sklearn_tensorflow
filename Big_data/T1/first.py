# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/6/5 19:19'

import numpy as np
import pandas as pd


## first
item_n = {"Negative":0,"Positive":0}
with open('sentimentLabel.txt','r') as f:
    for line in f:
        if line.split("\t")[0] == "Negative":
            item_n["Negative"] += 1
        if line.split("\t")[0] == "Positive":
            item_n["Positive"] += 1
print(item_n)
print("###################################################")
## second
negative = {}
positive = {}
with open('sentimentLabel.txt','r') as f:
    for line in f:
        if line.split("\t")[0] == "Negative":
            line = line.replace("\n","")
            for word in line.split(" ")[1:]:
                negative.setdefault(word,0)
                negative[word]+=1
        if line.split("\t")[0] == "Positive":
            line = line.replace("\n", "")
            for word in line.split(" ")[1:]:
                positive.setdefault(word,0)
                positive[word]+=1
negative = sorted(negative.items(),key=lambda item:item[1],reverse=True)
positive = sorted(positive.items(),key=lambda item:item[1],reverse=True)

print("Negative")
for key,value in negative[:10]:
    print(key+" "+str(value))
print("------------------------")
print("Positive")
for key,value in positive[:10]:
    print(key+" "+str(value))
print("###################################################")
############################3

## third
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


# *************************************************************************************************

