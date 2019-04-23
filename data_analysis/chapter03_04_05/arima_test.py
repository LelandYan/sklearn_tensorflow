# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/23 13:22'

if __name__ == '__main__':
    import pandas as pd

    discfile = './data/arima_data.xls'
    forecastnum = 5
    data = pd.read_excel(discfile, index_col=u'日期')

    import matplotlib.pyplot as plt

    # 正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 保存图像不能显示符号
    plt.rcParams['axes.unicode_minus'] = False

    # 时序图
    data.plot()
    # plt.show()

    # 自相关图 平稳性检测
    """平稳序列具有短期相关性，这个性质表示平稳序列通常只有
    近期的序列值对体现时值的影响比较明显，间隔较远的过去值对现值的影响的越小
    非平稳序列的自相关系数衰减的速度较慢
    """
    from statsmodels.graphics.tsaplots import plot_acf
    # plot_acf(data).show()
    # plt.show()

    # 平稳性检测
    from statsmodels.tsa.stattools import adfuller as ADF
    print("原始序列的ADF检验结果：",ADF(data['销量']))
    # 返回的值adf,pvalue,usedlag,nobs,critical values,icbest, regresults resstore

    # 差分后的结果
    D_data = data.diff().dropna()
    D_data.columns = ['销量差分']
    D_data.plot() # 时序图
    # plt.show()
    # plot_acf(D_data).show() # 自相关图

    from statsmodels.graphics.tsaplots import plot_pacf
    # plot_pacf(D_data).show() # 偏自相关图
    print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分']))  # 平稳性检测

    # 白噪声检验
    from statsmodels.stats.diagnostic import acorr_ljungbox

    print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值
    data[u'销量'] = data[u'销量'].astype(float)

    from statsmodels.tsa.arima_model import ARIMA

    pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # bic矩阵
    data.dropna(inplace=True)

    # 存在部分报错，所以用try来跳过报错；存在warning，暂未解决使用warnings跳过
    import warnings

    warnings.filterwarnings('error')
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    # 从中可以找出最小值
    bic_matrix = pd.DataFrame(bic_matrix)
    # 用stack展平，然后用idxmin找出最小值位置。
    p, q = bic_matrix.stack().idxmin()
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
    model = ARIMA(data, (p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型
    model.summary2()  # 给出一份模型报告
    model.forecast(forecastnum)  # 作为期5天的预测，返回预测结果、标准误差、置信区间。


