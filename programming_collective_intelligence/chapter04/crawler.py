# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/11 14:23'
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urljoin

ignorewords = {'the': 1, 'of': 1, 'to': 1, 'and': 1, 'a': 1, 'in': 1, 'is': 1, 'it': 1}
class clawler:
    # 初始化crawler类并传入数据库名称
    def __init__(self, dbname):
        pass

    def __del__(self):
        pass

    def dbcommit(self):
        pass

    # 辅助函数，用于获取条目ID，并且如果条目不存在，就将其加入数据库中
    def getentryid(self, table, field, value, createnew=True):
        return None 

    # 为这个网页建立索引
    def addtoindex(self, url, soup):
        print(f"Indexing {url}")

    # 从一个HTML文件中提取文字（不带标签）
    def gettextonly(self, soup):
        return None

    # 根据任何非空白字符进行分词处理
    def separatewords(self, text):
        return None

    # 如果url已经建立过索引，则返回true
    def isindexed(self, url):
        return True

    # 添加一个关联两个网页的链接
    def addlinkref(self, urlFrom, urlTo, linkText):
        pass

    # 从一个小组网页开始进行广度优先遍历搜索，直至达到某一深度
    # 期间为网页建立索引
    def crawl(self, pages, depth=2):
        pass

    # 建立数据库表
    def createindextables(self):
        pass
