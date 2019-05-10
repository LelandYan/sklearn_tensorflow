# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/9 21:59'

import time
import random
import math

people = [('Seymour', 'BOS'),
          ('Franny', 'DAL'),
          ('Zooey', 'CAK'),
          ('Walt', 'MIA'),
          ('Buddy', 'ORD'),
          ('Les', 'OMA')]
destination = 'LGA'
flights = {}


def getminutes(t):
    x = time.strptime(t, "%H:%M")
    return x[3] * 60 + x[4]


def printschedule(r):
    for d in range(len(r) // 2):
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin, destination)][r[2 * d]]
        ret = flights[(destination, origin)][r[2 * d + 1]]
        print("%10s%10s %5s-%5s $%3s %5s-%5s $%3s" % (name, origin, out[0], out[1], out[2], ret[0], ret[1], ret[2]))


for line in open("schedule.txt"):
    origin, dest, depart, arrive, price = line.strip().split(",")
    flights.setdefault((origin, dest), [])

    #
    flights[(origin, dest)].append((depart, arrive, int(price)))

s = [1, 4, 3, 2, 7, 3, 6, 3, 2, 4, 5, 3]


# printschedule(s)


def schedulecost(sol):
    totalprice = 0
    latestarrival = 0
    earliestdep = 24 * 60
    for d in range(len(sol) // 2):
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[2 * d])]
        returnf = flights[(destination, origin)][int(sol[2 * d + 1])]

        # 计算所有的去返航班的价格之和
        totalprice += outbound[2]
        totalprice += returnf[2]

        if latestarrival < getminutes(outbound[1]): latestarrival = getminutes(outbound[1])
        if earliestdep > getminutes(returnf[0]): earliestdep = getminutes(returnf[0])

    totalwait = 0
    for d in range(len(sol) // 2):
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[2 * d])]
        returnf = flights[(destination, origin)][int(sol[2 * d + 1])]
        totalwait += latestarrival - getminutes(outbound[1])
        totalwait += getminutes(returnf[0]) - earliestdep
    if latestarrival > earliestdep: totalprice += 50
    return totalwait + totalprice


# print(schedulecost(s))


# 随机搜索
def randomoptimize(domain, costf):
    best = 999999999
    bestr = None
    for i in range(1000):
        r = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        cost = costf(r)
        if cost < best:
            best = cost
            bestr = r
    return bestr


domain = [(0, 9)] * (len(people) * 2)
a = randomoptimize(domain, schedulecost)
cost = schedulecost(a)
print(cost)
printschedule(a)


# 爬山法
def hillclimb(domain, costf):
    # 创建一个随机解
    sol = [random.randint(domain[i][0], domain[i][0]) for i in range(len(domain))]

    while True:
        neighbors = []
        for j in range(len(domain)):
            if sol[j] > domain[j][0]:
                neighbors.append(sol[0:j] + [sol[j] - 1] + sol[j + 1:])
            if sol[j] < domain[j][1]:
                neighbors.append(sol[0:j] + [sol[j] + 1] + sol[j + 1:])
        current = costf(sol)
        best = current
        for j in range(len(neighbors)):
            cost = costf(neighbors[j])
            if cost < best:
                sol = neighbors[j]
            if best == current:
                break
        return sol


s1 = hillclimb(domain, schedulecost)
print(schedulecost(s1))
printschedule(s1)


# 模拟退火算法
def annealingoptimize(domain, costf, T=10000.0, cool=0.95, step=1):
    vec = [(random.randint(domain[i][0], domain[i][0])) for i in range(len(domain))]

    while T > 0.1:
        i = random.randint(0, len(domain) - 1)
        dir = random.randint(-step, step)
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]

        ea = costf(vec)
        eb = costf(vecb)

        if (eb < ea or random.random() < pow(math.e, -(eb - ea) / T)):
            vec = vecb

        T = T * cool
    return vec
s2 = annealingoptimize(domain,schedulecost)
print(s2)
print(schedulecost(s2))
printschedule(s2)

def geneticoptimize(domain,costf,popsize=50,step=1,mutprob=0.2,elite=0.2,maxiter=100):
    # 变异操作
    def mutate(vec):
        i = random.randint(0,len(domain)-1)
        if random.random() < 0.5 and vec[i] > domain[i][0]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]

    # 交叉操作
    def crossover(r1,r2):
        i = random.randint(1,len(domain)-2)
        return r1[0:i]+r2[i:]

    # 构造初始化种群
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        pop.append(vec)

    # 每一代中有多少胜出者
    topelite = int(elite*popsize)
    for i in range(maxiter):
        scores = [(costf(v),v) for v in pop]
        scores.sort()
        ranked = [v for (s,v) in scores]

        pop = ranked[0:topelite]

        while len(pop) < popsize:
            if random.random() < mutprob:
                c = random.randint(0,topelite)
                pop.append(mutate(ranked[c]))
            else:
                c1 =random.randint(0,topelite)
                c2 = random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2]))
        # print(scores[0][0])
    return scores[0][1]

s3 = geneticoptimize(domain,schedulecost)
printschedule(s3)
# printschedule(s3)