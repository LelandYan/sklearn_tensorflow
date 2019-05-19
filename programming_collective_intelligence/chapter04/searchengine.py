# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/10 16:21'
import urllib
from urlparse import urljoin
from bs4 import BeautifulSoup
import sqlite3
import re

ignorewords = {'the', 'of', 'to', 'and', 'a', 'in', 'is', 'it'}


class crawler:
    # 初始化crawle类并传入数据库名称
    def __init__(self, dbname):
        self.con = sqlite3.connect(dbname)

    def __del__(self):
        self.con.close()

    def dbcommit(self):
        self.con.commit()

    # 用于获取条目的id，并且如果条目不存在，就将其加入数据库中
    def getentryid(self, table, field, value, createnew=True):
        cur = self.con.execute("select rowid from %s where %s='%s'" % (table, field, value))
        res = cur.fetchone()
        if res == None:
            cur = self.con.execute("insert into %s (%s) values ('%s')" % (table, field, value))
            return cur.lastrowid
        else:
            return res[0]

            # 为每个网页建立索引
    def addtoindex(self, url, soup):
        if self.isindexed(url): return
        print('Indexing {}'.format(url))

        # 获取每个单词
        text = self.gettextonly(soup)
        words = self.separatewords(text)

        # 得到URL的id
        urlid = self.getentryid('urllist', 'url', url)

        # 将每个单词与该url关联
        for i in range(len(words)):
            word = words[i]
            if word in ignorewords: continue
            wordid = self.getentryid('wordlist', 'word', word)
            self.con.execute(
                "insert into wordlocation(urlid,wordid,location) values ({},{},{})".format(urlid, wordid, i))

    # 从一个HTML网页中提取文字（不带标签）
    def gettextonly(self,  soup):
        v = soup.string
        if v == None:
            c = soup.contents
            resulttext = ""
            for t in c:
                subtext = self.gettextonly(t)
                resulttext += subtext + "\n"
            return resulttext
        else:
            return v.strip()

    # 根据任何非空白字符进行分词处理
    def separatewords(self, text):
        splitter = re.compile('\\W*')
        return [s.lower() for s in splitter.split(text) if s != '']

    # 如果url已经建立索引，返回Ture
    def isindexed(self, url):
        u = self.con.execute("select rowid from urllist where url='%s'" % url).fetchone()
        if u != None:
            v = self.con.execute("select * from wordlocation where urlid=%d" % u[0]).fetchone()
            if v != None:
                return True
        return False

    # 添加一个关联两个网页的连接
    def addlinkref(self, urlFrom, urlTo, linkText):
        pass

    # 从一小组网页开始进行广度优先搜索，直到一定的深度
    # 期间为网页建立索引
    def crawl(self, pages, depth=2):
        for i in range(depth):
            print("depth {} begins".format(i))
            new_pages = set()
            for page in pages:
                try:
                    c = urllib.urlopen(page)
                except:
                    print("Could not open {}".format(page))
                    continue
                soup = BeautifulSoup(c.read(), 'lxml')
                #self.addtoindex(page, soup)
                print(page)
                links = soup('a')
                for link in links:
                    if ('href' in dict(link.attrs)):
                        url = urljoin(page, link['href'])
                        if url.find("'") != -1:
                            continue
                        url = url.split('#')[0]  # remove location portion
                        if url[0:4] == 'http' and not self.isindexed(url):
                            new_pages.add(url)
                        link_text = self.gettextonly(link)
                        self.addlinkref(page, url, link_text)
                #self.dbcommit()
            pages = new_pages

    # 创建数据库表
    def createindextables(self):
        self.con.execute("create table urllist(url)")
        self.con.execute('create table wordlist(word)')
        self.con.execute('create table wordlocation(urlid,wordid,location)')
        self.con.execute('create table link(fromid integer,toid integer)')
        self.con.execute('create table linkwords(wordid,linkid)')
        self.con.execute('create index wordidx on wordlist(word)')
        self.con.execute('create index urlidx on urllist(url)')
        self.con.execute('create index wordurlidx on wordlocation(wordid)')
        self.con.execute('create index urltoidx on link(toid)')
        self.con.execute('create index urlfromidx on link(fromid)')
        self.dbcommit()


page_list = ['https://github.com/explore']
crawler = crawler("searchindex.db")
# crawler.createindextables()
crawler.crawl(page_list)
