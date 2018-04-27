# -*- coding: utf-8 -*-

"""LightGBM dataset divided by market.
IMPORTANT:
To run this model you need run before preprocessing/preprocessing.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from utils import *

np.random.seed(0)

# Settings
SUBMISSION = True # True to create csv to submission; False to local validation
rmse_accumulated = 0.0
y_pred_final = None


df_train_general = pd.read_csv('../input/train_fe.csv', index_col=0)
df_train_general = fillna_bystock(df_train_general)

if SUBMISSION:
    df_test_general = pd.read_csv('../input/test_fe.csv', index_col=0)
    df_test_general = fillna_bystock(df_test_general)
    df_test_final = pd.read_csv('../input/test.csv', index_col=0)

for i in [1,2,3,4]:
    print("\n \nAnalyze Market: %i \n" % i)
    df_train = df_train_general[df_train_general['Market'] == i]

    excl = ['Weight', 'y', 'Market', 'Date']
    cols = [c for c in df_train.columns if c not in excl]

    if SUBMISSION:
        df_test = df_test_general[df_test_general['Market'] == i]
        # Split
        X_train = df_train[cols]
        y_train = df_train.y
        X_test = df_test[cols]
    else:
        # X_train, X_test, y_train, y_test = train_test_split(df_train[cols], df_train['y'], test_size=0.5, random_state=17)
        X_train, X_test, y_train, y_test = train_test_split_own(df_train[cols], df_train['y'])

    # Prepare data to lightgbm
    lgb_train = lgb.Dataset(X_train[cols], y_train)
    # lgb_eval = lgb.Dataset(X_test[cols], y_test, reference=lgb_train)
    # Params
    if i == 1:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'mse'},
            'num_leaves': 127,
            'learning_rate': 0.05,
            'bagging_fraction': 0.9, # 1
            'feature_fraction': 0.95, # 1
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 350, # default=20
            'verbose': 0
        }
    elif i == 2:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'mse'},
            'num_leaves': 127,
            'learning_rate': 0.05,
            'bagging_fraction': 0.9, # 1
            'feature_fraction': 0.95, # 1
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 350, # default=20
            'verbose': 0
        }
    elif i == 3:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'mse'},
            'num_leaves': 127,
            'learning_rate': 0.05,
            'bagging_fraction': 0.95, # 1
            'feature_fraction': 1, # 1
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 160, # default=20
            'verbose': 0
        }
    elif i == 4:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'mse'},
            'num_leaves': 127,
            'learning_rate': 0.05,
            'bagging_fraction': 0.9, # 1
            'feature_fraction': 0.95, # 1
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 160, # default=20
            'verbose': 0
        }

    bst_lst = []
    for idx in range(8):
        params['seed'] = 2429 + 513 * idx
        # Train with cv
        bst_lst.append(lgb.train(params,
                        lgb_train,
                        num_boost_round=800))
                        #valid_sets=lgb_eval,
                        #early_stopping_rounds=20)

    # Predict
    pred_list = []
    for bst in bst_lst:
        pred_list.append(bst.predict(X_test, num_iteration=bst.best_iteration))
    y_pred = np.array(pred_list).mean(0)

    # Concatenate predictions
    if i == 1:
        y_pred_final = y_pred
    else:
        y_pred_final = np.hstack((y_pred_final, y_pred))

    if not SUBMISSION:
        rmse_accumulated += (sklearn.metrics.mean_squared_error(y_test.values, y_pred) * y_test.count())
        print(sklearn.metrics.mean_squared_error(y_test.values, y_pred) * y_test.count())

if SUBMISSION:
    # to submission
    df_test_final['y'] = y_pred_final
    df_test_final['y'].to_csv('submission42.csv', header=True)
    print(".csv submission is ready")
else:
    print("\n\nRMSE TOTAL: " + str(rmse_accumulated))
