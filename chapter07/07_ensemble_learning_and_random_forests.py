# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/21 9:24'

import numpy as np
import matplotlib.pyplot as plt

# heads_proba = 0.51
# coin_tosses = (np.random.rand(10000,10) < heads_proba).astype(np.int32)
# print(coin_tosses)
# cumulative_heads_ratio = np.cumsum(coin_tosses,axis=0) / np.arange(1,10001).reshape(-1,1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# hard voting
# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf = SVC(random_state=42)
#
# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#     voting='hard')
#
# from sklearn.metrics import accuracy_score
#
# for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
#     clf.fit(X_train,y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__,accuracy_score(y_test,y_pred))
#
# # soft voting
# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf = SVC(probability=True, random_state=42)
# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#     voting='soft')
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# Bagging ensembles
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(random_state=42), n_estimators=500, max_samples=100, bootstrap=True, random_state=42
# )
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# from sklearn.metrics import accuracy_score
#
# print("使用bagging的决策树：",accuracy_score(y_test, y_pred))
#
# tree_clf = DecisionTreeClassifier(random_state=42)
# tree_clf.fit(X_train, y_train)
# y_pred_tree = tree_clf.predict(X_test)
# print("决策树：",accuracy_score(y_test, y_pred_tree))

from matplotlib.colors import ListedColormap


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=False):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


# plt.figure(figsize=(11, 4))
# plt.subplot(121)
# plot_decision_boundary(tree_clf, X, y)
# plt.title("Decision Tree", fontsize=14)
# plt.subplot(122)
# plot_decision_boundary(bag_clf, X, y)
# plt.title("Decision Trees with Bagging", fontsize=14)
# plt.show()


# # 随机森林 （使用Bagging实现）
# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(splitter='random',max_leaf_nodes=16,random_state=42),
#     n_estimators=500,max_samples=1.0,bootstrap=True,random_state=42
# )
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# print(accuracy_score(y_pred,y_test))
#
# # 使用随机森林库
# from sklearn.ensemble import RandomForestClassifier
# rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
# rnd_clf.fit(X_train, y_train)
# y_pred_rf = rnd_clf.predict(X_test)
# print(accuracy_score(y_pred_rf,y_test))


from sklearn.datasets import load_iris

# iris = load_iris()
# rnd_clf = RandomForestClassifier(n_estimators=500,random_state=42)
# rnd_clf.fit(iris['data'],iris['target'])
# for name,score in zip(iris['feature_names'],rnd_clf.feature_importances_):
#     print(name,score)

# Out of Bag evaluation
# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(random_state=42), n_estimators=500,
#     bootstrap=True, oob_score=True, random_state=40)
# bag_clf.fit(X_train, y_train)
# print(bag_clf.oob_score_)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

# ada_clf = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=1),n_estimators=200,
#     algorithm='SAMME.R',learning_rate=0.5,random_state=42
# )
# ada_clf.fit(X_train,y_train)
# plot_decision_boundary(ada_clf,X,y)
# plt.show()

#
# m = len(X_train)
# plt.figure(figsize=(11,4))
# for subplot,learning_rate in((121,1),(122,0.5)):
#     sample_weights = np.ones(m)
#     plt.subplot(subplot)
#     for i in range(5):
#         svm_clf = SVC(kernel="rbf", C=0.05, random_state=42)
#         svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
#         y_pred = svm_clf.predict(X_train)
#         sample_weights[y_pred != y_train] *= (1 + learning_rate)
#         plot_decision_boundary(svm_clf,X,y,alpha=0.2)
#         plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
#     if subplot == 121:
#         plt.text(-0.7, -0.65, "1", fontsize=14)
#         plt.text(-0.6, -0.10, "2", fontsize=14)
#         plt.text(-0.5, 0.10, "3", fontsize=14)
#         plt.text(-0.4, 0.55, "4", fontsize=14)
#         plt.text(-0.3, 0.90, "5", fontsize=14)

# plt.show()
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
# gbrt_slow.fit(X, y)

# Gradient Boosting with Early stopping
from sklearn.metrics import mean_squared_error
# X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
# gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
# gbrt.fit(X_train, y_train)
#
# errors = [mean_squared_error(y_val, y_pred)
#           for y_pred in gbrt.staged_predict(X_val)]
# bst_n_estimators = np.argmin(errors)
#
# gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
# gbrt_best.fit(X_train, y_train)
#
# from sklearn.linear_model import LogisticRegression
# log_clf = LogisticRegression().fit(X_train,y_train)

from xgboost import XGBRegressor

# xgb_reg = XGBRegressor(random_state=42)
# xgb_reg.fit(X_train, y_train)
# y_pred = xgb_reg.predict(X_val)
# val_error = mean_squared_error(y_val, y_pred)
# print("Validation MSE:", val_error)
# xgb_reg.fit(X_train, y_train,
#                 eval_set=[(X_val, y_val)], early_stopping_rounds=2)
# y_pred = xgb_reg.predict(X_val)
# val_error = mean_squared_error(y_val, y_pred)  # Not shown
# print("Validation MSE:", val_error)

from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# 导入数据
mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the:", estimator)
    estimator.fit(X_train, y_train)

validation_res = [estimator.score(X_val, y_val) for estimator in estimators]
print(validation_res)

from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf)
]
voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
print("硬投票的结果:", voting_clf.score(X_val, y_val))

# 将投票的分类器中的svm置空,但是依然存在
voting_clf.set_params(svm_clf=None)
print(voting_clf.estimators)
print(voting_clf.estimators_)
# 删除svm分类器
del voting_clf.estimators_[2]
print(voting_clf.score(X_val, y_val))
# 采用软分类
voting_clf.voting = "soft"
print(voting_clf.score(X_val, y_val))
print(voting_clf.score(X_test, y_test))
res = [estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]
print(res)

# Stacking
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)
print(rnd_forest_blender.oob_score_)

X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
y_pred = rnd_forest_blender.predict(X_test_predictions)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
