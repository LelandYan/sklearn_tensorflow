# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/6/5 20:53'

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