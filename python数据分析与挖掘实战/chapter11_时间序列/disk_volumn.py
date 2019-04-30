# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/26 10:47'

import pandas as pd


def attribute_transform():
    discfile = './data/discdata.xls'
    transformeddata = './tmp/discdata_processed.xls'

    data = pd.read_excel(discfile)
    data = data[data['TARGET_ID'] == 184].copy()

    data_group = data.groupby('COLLECTTIME')

    def attr_trans(x):
        result = pd.Series(
            index=["SYS_NAME", "CWXT_DB:184:C:\\", "CWXT_DB:184:D:\\", "COLLECTTIME"])
        result["SYS_NAME"] = x["SYS_NAME"].iloc[0]
        result["COLLECTTIME"] = x["COLLECTTIME"].iloc[0]
        result["CWXT_DB:184:C:\\"] = x["VALUE"].iloc[0]
        result["CWXT_DB:184:D:\\"] = x["VALUE"].iloc[1]

        return result
    data_processed = data_group.apply(attr_trans)
    data_processed.to_excel(transformeddata,index=False)


def stationarity_test():
    discfile = './data/discdata_processed.xls'

    data = pd.read_excel(discfile)
    data = data.iloc[:len(data)-5]

    from statsmodels.tsa.stattools import adfuller as ADF
    diff = 0
    adf = ADF(data["CWXT_DB:184:D:\\"])
    while adf[1] >= 0.05:
        diff +=1
        adf = ADF(data["CWXT_DB:184:D:\\"].diff(diff).dropna())

    print("原始序列经过{}阶差分后归于平稳,对应的p值为{}".format(diff,adf[1]))

def whitenoise_test():
    from statsmodels.stats.diagnostic import acorr_ljungbox
    discfile = "./data/discdata_processed.xls"

    data = pd.read_excel(discfile)
    data = data.iloc[:len(data) - 5]

    [[lb], [p]] = acorr_ljungbox(data["CWXT_DB:184:D:\\"], lags=1)
    if p < 0.05:
        print(u"原始序列为非白噪声序列，对应的p值为：%s" % p)
    else:
        print(u"原始序列为白噪声序列，对应的p值为：%s" % p)

    [[lb], [p]] = acorr_ljungbox(
        data["CWXT_DB:184:D:\\"].diff().dropna(), lags=1)

    if p < 0.05:
        print(u"一阶差分序列为非白噪声序列，对应的p值为：%s" % p)
    else:
        print(u"一阶差分序列为白噪声序列，对应的p值为：%s" % p)
    print(lb)

def find_optimal_pq():
    discfile = "data/discdata_processed.xls"

    data = pd.read_excel(discfile, index_col="COLLECTTIME")
    # 不使用最后五个数据
    data = data.iloc[:len(data) - 5]
    xdata = data["CWXT_DB:184:D:\\"]

    from statsmodels.tsa.arima_model import ARIMA

    pmax = int(len(xdata) / 10)
    qmax = int(len(xdata) / 10)

    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(xdata, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    # 找出最小值
    p, q = bic_matrix.stack().idxmin()
    print(u"BIC最小的p值和q值为：%s、%s" % (p, q))

def arima_model_check():
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    discfile = "data/discdata_processed.xls"
    # 残差延迟个数
    lagnum = 12

    data = pd.read_excel(discfile, index_col="COLLECTTIME")
    data = data.iloc[:len(data) - 5]
    xdata = data["CWXT_DB:184:D:\\"]

    # 训练模型并预测，计算残差
    arima = ARIMA(xdata, (0, 1, 1)).fit()
    xdata_pred = arima.predict(typ="levels")
    pred_error = (xdata_pred - xdata).dropna()

    lb, p = acorr_ljungbox(pred_error, lags=lagnum)
    h = (p < 0.05).sum()
    if h > 0:
        print(u"模型ARIMA（0,1,1)不符合白噪声检验")
    else:
        print(u"模型ARIMA（0,1,1)符合白噪声检验")
    print(lb)

def cal_errors():
    file = "data/predictdata.xls"
    data = pd.read_excel(file)

    # 计算误差
    abs_ = (data[u"预测值"] - data[u"实际值"]).abs()
    mae_ = abs_.mean()
    rmse_ = ((abs_ ** 2).mean()) ** 0.5
    mape_ = (abs_ / data[u"实际值"]).mean()

    print(u"平均绝对误差为：%0.4f, \n 均方根误差为%0.4f, \n平均绝对百分误差为：%0.6f。" % (mae_, rmse_, mape_))
if __name__ == '__main__':
    # attribute_transform()
    # stationarity_test()
    # whitenoise_test()
    find_optimal_pq()