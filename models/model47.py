# -*- coding: utf-8 -*-

"""XGBoost full dataset.
IMPORTANT:
To run this model you need run before preprocessing/preprocessing.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import operator

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from utils import *

np.random.seed(0)

# Settings
OUTLIERS = False
OUTLIER_FACTOR = 10
SUBMISSION = True # True to create csv to submission; False to local validation
rmse_accumulated = 0.0

# Load data
df_train = pd.read_csv('../input/train_fe.csv', index_col=0)
# Fill nan
df_train = fillna_bystock(df_train)

excl = ['Weight', 'y', 'Date']
cols = [c for c in df_train.columns if c not in excl]

if SUBMISSION:
    # Load test data
    df_test = pd.read_csv('../input/test_fe.csv', index_col=0)
    df_test = fillna_bystock(df_test)

    # Split
    X_train = df_train[cols]
    y_train = df_train.y
    X_test = df_test[cols]
else:
    X_train, X_test, y_train, y_test = train_test_split_own(df_train[cols], df_train.y)

# Delete Outliers
if OUTLIERS:
    X_train, y_train = delete_outliers(X_train, y_train, OUTLIER_FACTOR)
    X_test = clip_outliers(X_test, OUTLIER_FACTOR)

xgmat_train = xgb.DMatrix(X_train[cols], y_train)
xgmat_test = xgb.DMatrix(X_test[cols])

params_xgb = {
    'objective'        : 'reg:linear',
    'tree_method'      : 'hist',
    'grow_policy'      : 'depthwise',
    'eta'              : 0.1,
    'max_depth'        : 30,
    'min_child_weight' : 160,
    'subsample'        : 0.95,
    'colsample_bytree' : 1,
    'colsample_bylevel': 1,
    'base_score'       : y_train.mean(),
    'eval_metric'      : 'rmse',
    'silent'           : True
}

n_round = 42

"""
# cross-validation
cv_result = xgb.cv(params_xgb,
                   xgmat_train,
                   num_boost_round=1000,
                   early_stopping_rounds=50,
                   verbose_eval=50,
                   show_stdv=False)
n_round = len(cv_result)
"""

bst_lst = []
for idx in range(8):
    params_xgb['seed'] = 2429 + 513 * idx
    bst_lst.append(xgb.train(params_xgb, xgmat_train, num_boost_round=n_round))

# Predict
pred_list = []
for bst in bst_lst:
    pred_list.append(bst.predict(xgmat_test))

y_pred = np.array(pred_list).mean(0)

# Results
if SUBMISSION:
    # to submission
    df_test['y'] = y_pred
    df_test['y'].to_csv('submission47.csv', header=True)
    print(".csv submission is ready")
else:
    # metric
    rmse_accumulated += sklearn.metrics.mean_squared_error(y_test.values, y_pred) * y_test.count()
    print(str(sklearn.metrics.mean_squared_error(y_test.values, y_pred) * y_test.count()))
