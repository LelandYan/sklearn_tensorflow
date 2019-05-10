# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/10 12:44'

import random
import math
import numpy as np
# The dorms, each of which has two available spaces
dorms = ['Zeus', 'Athena', 'Hercules', 'Bacchus', 'Pluto']

# People, along with their first and second choices
prefs = [('Toby', ('Bacchus', 'Hercules')),
         ('Steve', ('Zeus', 'Pluto')),
         ('Karen', ('Athena', 'Zeus')),
         ('Sarah', ('Zeus', 'Pluto')),
         ('Dave', ('Athena', 'Bacchus')),
         ('Jeff', ('Hercules', 'Pluto')),
         ('Fred', ('Pluto', 'Athena')),
         ('Suzie', ('Bacchus', 'Hercules')),
         ('Laura', ('Bacchus', 'Hercules')),
         ('James', ('Hercules', 'Athena'))]

domain = [(0, (len(dorms) * 2) - i - 1) for i in range(0, len(dorms) * 2)]


def printsolution(vec):
    slots = []
    # Create two slots for each dorm
    for i in range(len(dorms)): slots += [i, i]

    # Loop over each students assignment
    for i in range(len(vec)):
        x = int(vec[i])

        # Choose the slot from the remaining ones
        dorm = dorms[slots[x]]
        # Show the student and assigned dorm
        print(prefs[i][0], dorm)
        # Remove this slot
        del slots[x]


def dormcost(vec):
    cost = 0
    # Create list a of slots
    slots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    # Loop over each student
    for i in range(len(vec)):
        x = int(vec[i])
        dorm = dorms[slots[x]]
        pref = prefs[i][1]
        # First choice costs 0, second choice costs 1
        if pref[0] == dorm:
            cost += 0
        elif pref[1] == dorm:
            cost += 1
        else:
            cost += 3
        # Not on the list costs 3

        # Remove selected slot
        del slots[x]

    return cost


def geneticoptimize(domain, costf, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
    # 变异操作
    def mutate(vec):
        i = random.randint(0, len(domain) - 1)
        if random.random() < 0.5 and vec[i] > domain[i][0]:
            return vec[0:i] + [vec[i] - step] + vec[i + 1:]
        elif vec[i] < domain[i][1]:
            return vec[0:i] + [vec[i] + step] + vec[i + 1:]

    # 交叉操作
    def crossover(r1, r2):
        i = random.randint(1, len(domain) - 2)
        return r1[0:i] + r2[i:]

    # 构造初始化种群
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        pop.append(vec)

    # 每一代中有多少胜出者
    scores = None
    topelite = int(elite * popsize)
    bestPop = pop[0:topelite]
    for i in range(maxiter):
        try:
            scores = [(costf(v), v) for v in pop]
        except:
            pass
        scores.sort()
        ranked = [v for (s, v) in scores]
        # print("@@",np.sum(np.array(bestPop)))
        # print("###",np.sum(np.array(ranked[0:topelite])))
        # if (np.sum(np.array(bestPop)) == np.sum(np.array(ranked[0:topelite]))):
        #     break
        # 选出这次迭代的最优个体
        pop = ranked[0:topelite]
        bestPop = pop


        while len(pop) < popsize:
            if random.random() < mutprob:
                c = random.randint(0, topelite)
                pop.append(mutate(ranked[c]))
            else:
                c1 = random.randint(0, topelite)
                c2 = random.randint(0, topelite)
                pop.append(crossover(ranked[c1], ranked[c2]))
        # print(scores[0][0])

    return scores[0][1]


def annealingoptimize(domain, costf, T=1000000.0, cool=0.95, step=1):
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


s1 = randomoptimize(domain, dormcost)
print(dormcost(s1))
s2 = annealingoptimize(domain, dormcost)
print(dormcost(s2))
s3 = hillclimb(domain, dormcost)
print(dormcost(s3))
s4 = geneticoptimize(domain, dormcost)
print(dormcost(s4))
