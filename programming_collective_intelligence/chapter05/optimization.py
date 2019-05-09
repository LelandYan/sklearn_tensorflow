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

flights = {}
for line in open("schedule.txt"):
    origin,dest,depart,arrive,price = line.strip().split(",")
    flights.setdefault((origin,dest),[])

    #
    flights[(origin,dest)].append((depart,arrive,int(price)))
