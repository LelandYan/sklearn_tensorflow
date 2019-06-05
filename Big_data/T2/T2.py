# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/6/5 20:54'

from sklearn.datasets import load_svmlight_files
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
files = ["aclImdb/train/labeledBow.feat", "aclImdb/test/labeledBow.feat"]
training_data, raw_training_target, testing_data, raw_testing_target = load_svmlight_files(files, n_features=None,
                                                                                           dtype=None)

tf_transformer = TfidfTransformer()
# It computes the TF for each review, the IDF using each review, and finally the TF-IDF for each review
training_data_tfidf = tf_transformer.fit_transform(training_data)
# .transform on the testing data which computes the TF for each review,
# then the TF-IDF for each review using the IDF from the training data
testing_data_tfidf = tf_transformer.transform(testing_data)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(training_data_tfidf, raw_training_target)
predict = clf.predict(testing_data_tfidf)
from sklearn import metrics

print(metrics.classification_report(predict, raw_testing_target))
