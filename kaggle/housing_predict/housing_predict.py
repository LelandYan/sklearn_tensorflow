# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/17 19:03'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读入数据
train_df = pd.read_csv("./input/train.csv", index_col=0)
test_df = pd.read_csv("./input/test.csv", index_col=0)

# 可视化并平滑化要求的数据
# prices = pd.DataFrame({"prices": train_df["SalePrice"], "log(price+1)": np.log1p(train_df["SalePrice"])})
# prices.hist()
# plt.show()

y_train = np.log1p(train_df.pop("SalePrice"))

all_df = pd.concat((train_df,test_df),axis=0)

all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

# print(all_df['MSSubClass'].value_counts())
# print(pd.get_dummies(all_df["MSSubClass"],prefix="MSSubClass").head())

all_dummy_df = pd.get_dummies(all_df)

# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))

mean_cols = all_dummy_df.mean()

all_dummy_df = all_dummy_df.fillna(mean_cols)
# print(all_dummy_df.isnull().sum().sum())

numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std


dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

# print(dummy_train_df.shape, dummy_test_df.shape)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

X_train = dummy_train_df.values
X_test = dummy_test_df.values

# alphas = np.logspace(-3,2,50)
# test_scores = []
# for alpha in alphas:
#     clf = Ridge(alpha)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring="neg_mean_squared_error"))
#     test_scores.append(np.mean(test_score))

# print(test_scores)
# plt.plot(alphas, test_scores,"r-")
# plt.title("Alpha vs CV Error")
# plt.show()

# from sklearn.ensemble import RandomForestRegressor
# max_features = [.1, .3, .5, .7, .9, .99]
# test_scores = []
# for max_feat in max_features:
#     clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))

# plt.plot(max_features, test_scores)
# plt.title("Max Features vs CV Error")
# plt.show()

from xgboost import XGBRegressor

# params = [1,2,3,4,5,6]
# test_scores = []
# min_param = -1
# for param in params:
#     clf = XGBRegressor(max_depth=param)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring="neg_mean_squared_error"))
#     test_scores.append(np.mean(test_score))
#
#
# plt.plot(params,test_scores)
# plt.title("max_depth vs CV Error")
# plt.show()
clf = XGBRegressor(max_depth=5)
clf.fit(X_train,y_train)
y_final = clf.predict(X_test)

submission_df = pd.DataFrame(data={"Id":test_df.index,"SalePrice":y_final})
submission_df.to_csv("result.csv",index=0)