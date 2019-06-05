# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/6/5 20:53'

item_n = {"Negative":0,"Positive":0}
with open('sentimentLabel.txt','r') as f:
    for line in f:
        if line.split("\t")[0] == "Negative":
            item_n["Negative"] += 1
        if line.split("\t")[0] == "Positive":
            item_n["Positive"] += 1
print(item_n)