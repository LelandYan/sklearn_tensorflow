# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/10 13:37'

import math
import random
people = ['Charlie', 'Augustus', 'Veruca', 'Violet', 'Mike', 'Joe', 'Willy', 'Miranda']

links = [('Augustus', 'Willy'),
         ('Mike', 'Joe'),
         ('Miranda', 'Mike'),
         ('Violet', 'Augustus'),
         ('Miranda', 'Willy'),
         ('Charlie', 'Mike'),
         ('Veruca', 'Joe'),
         ('Miranda', 'Augustus'),
         ('Willy', 'Augustus'),
         ('Joe', 'Charlie'),
         ('Veruca', 'Augustus'),
         ('Miranda', 'Joe')]


def crosscount(v):
    # Convert the number list into a dictionary of person:(x,y)
    loc = dict([(people[i], (v[i * 2], v[i * 2 + 1])) for i in range(0, len(people))])
    total = 0

    # Loop through every pair of links
    for i in range(len(links)):
        for j in range(i + 1, len(links)):

            # Get the locations
            (x1, y1), (x2, y2) = loc[links[i][0]], loc[links[i][1]]
            (x3, y3), (x4, y4) = loc[links[j][0]], loc[links[j][1]]

            den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

            # den==0 if the lines are parallel
            if den == 0: continue

            # Otherwise ua and ub are the fraction of the
            # line where they cross
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

            # If the fraction is between 0 and 1 for both lines
            # then they cross each other
            if 0 < ua < 1 and 0 < ub < 1:
                total += 1
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                # Get the locations of the two nodes
                (x1, y1), (x2, y2) = loc[people[i]], loc[people[j]]

                # Find the distance between them
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
                # Penalize any nodes closer than 50 pixels
                if dist < 50:
                    total += (1.0 - (dist / 50.0))

    return total


from PIL import Image, ImageDraw


def drawnetwork(sol):
    # Create the image
    img = Image.new('RGB', (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Create the position dict
    pos = dict([(people[i], (sol[i * 2], sol[i * 2 + 1])) for i in range(0, len(people))])

    for (a, b) in links:
        draw.line((pos[a], pos[b]), fill=(255, 0, 0))

    for n, p in pos.items():
        draw.text(p, n, (0, 0, 0))

    img.save("123.png")
    #img.show()


domain = [(10, 370)] * (len(people) * 2)



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




sol1 = randomoptimize(domain,crosscount)
sol2 = geneticoptimize(domain,crosscount)
sol3 = annealingoptimize(domain,crosscount,step=20,T=100000,cool=0.999)
drawnetwork(sol3)