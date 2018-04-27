# -*- coding: utf-8 -*-

"""XGBoost dataset divided by market.
IMPORTANT:
To run this model you need run before preprocessing/preprocessing.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from utils import *

"""
IMPORTANT:
To run this model you need run before preprocessing/preprocessing_xx.py
"""

np.random.seed(0)

# Settings
OUTLIERS = False
OUTLIER_FACTOR = 6
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

    print("Number of columns: " + str(len(X_train.columns)))

    # Delete Outliers
    if OUTLIERS:
        X_train, y_train = delete_outliers(X_train, y_train, OUTLIER_FACTOR)
        X_test = clip_outliers(X_test, OUTLIER_FACTOR)

    # Prepare datas to xgboost
    xgmat_train = xgb.DMatrix(X_train[cols], y_train)
    xgmat_test = xgb.DMatrix(X_test[cols])

    # Params
    params_xgb = {'objective'        : 'reg:linear',
                  'tree_method'      : 'hist',
                  'grow_policy'      : 'depthwise',
                  'eta'              : 0.05,
                  'max_depth'        : 30,
                  'min_child_weight' : 160,
                  #'gamma'            : 0.00005,
                  'subsample'        : 0.95,
                  'colsample_bytree' : 1,
                  'colsample_bylevel': 1,
                  'base_score'       : y_train.mean(),
                  'eval_metric'      : 'rmse',
                  'silent'           : True
    }
    n_round = 300

    # Cross-Validation
    """
    cv_result = xgb.cv(params_xgb,
                   xgmat_train,
                   num_boost_round=500,
                   early_stopping_rounds=20,
                   verbose_eval=20,
                   show_stdv=False)
    n_round = len(cv_result)
    """

    # Train
    bst_lst = []
    for idx in range(8):
        params_xgb['seed'] = 2429 + 513 * idx
        bst_lst.append(xgb.train(params_xgb, xgmat_train, num_boost_round=n_round))

    # Predict
    pred_list = []
    for bst in bst_lst:
        pred_list.append(bst.predict(xgmat_test))

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
    df_test_final['y'].to_csv('submission45.csv', header=True)
    print(".csv submission is ready")
else:
    print("\n\nRMSE TOTAL: " + str(rmse_accumulated))
