#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:22:57 2018

@author: hjh
"""
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from scipy import sparse
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def KFold_test(train_X, train_y, test):
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=False, random_state=42)

    test_preds = np.zeros((test.shape[0], N))
    train_preds = np.zeros(train_X.shape[0])

    for k, (train_in, test_in) in enumerate(skf.split(train_X, train_y)):
        print('第%s折' % k)
        X_train, X_valid, y_train, y_valid = train_X.iloc[train_in], train_X.iloc[test_in], train_y[train_in], train_y[test_in]
        print(X_train.shape)
        print(X_valid.shape)
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=1500, objective='binary',
            subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
            learning_rate=0.01, min_child_weight=40, random_state=2018,
            n_jobs=100, verbose=0
        )
        clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc', early_stopping_rounds=100)
        print('Start predicting...')
        train_preds[test_in] += clf.predict_proba(X_valid)[:,1]
        test_preds[:, k] = clf.predict_proba(test)[:,1]
    print('线下成绩约', roc_auc_score(train_y, train_preds))

if __name__ == '__main__':
    train_agg = pd.read_table('..\data\\train_agg.csv')
    train_flag = pd.read_table('..\data\\train_flg.csv')
    test_agg = pd.read_table('..\data\\test_agg.csv')

    print('origin feature:')
    KFold_test(train_agg, train_flag['FLAG'], test_agg)

