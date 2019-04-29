# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/29 21:00'

import pandas as pd
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')
engine = create_engine(
        r"mysql+pymysql://root:yanxiangpei123@127.0.0.1:3306/test?charset=utf8")
sql = pd.read_sql("all_gzdata", engine, chunksize=10000)
counts = [ i['fullURLId'].value_counts() for i in sql] #逐块统计
counts = pd.concat(counts).groupby(level=0).sum() #合并统计结果，把相同的统计项合并（即按index分组并求和）
counts = counts.reset_index() #重新设置index，将原来的index作为counts的一列。
counts.columns = ['index', 'num'] #重新设置列名，主要是第二列，默认为0
counts['type'] = counts['index'].str.extract('(\d{3})') #提取前三个数字作为类别id
counts['percent'] = counts['num']/counts['num'].sum()*100
counts_ = counts[['type', 'num','percent']].groupby('type').sum() #按类别合并
counts_.sort_values('num', ascending = False) #降序排列
print(counts_)
