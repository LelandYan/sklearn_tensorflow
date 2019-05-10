# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/10 12:22'

import xml.dom.minidom
dom = xml.dom.minidom.parseString('<data><rec>Hello!</rec></data>')
print(dom)