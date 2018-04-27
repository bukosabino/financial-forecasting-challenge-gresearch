# -*- coding: utf-8 -*-

"""LightGBM full dataset.
IMPORTANT:
To run this model you need run before preprocessing/preprocessing.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from datetime import timedelta
import operator

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from utils import *

np.random.seed(0)

# Settings
OUTLIERS = False
OUTLIER_FACTOR = 6
SUBMISSION = True # True to create csv to submission; False to local validation
N_CV = 10
rmse_accumulated = 0.0

# Load data
df_train = pd.read_csv('../input/train_fe.csv', index_col=0)

# Fill nan
df_train = fillna_bystock(df_train)

excl = ['Weight', 'y', 'Date']
cols = [c for c in df_train.columns if c not in excl]

#for i in range(N_CV+1):

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
    """
    if i == 0:
        X_train, X_test, y_train, y_test = train_test_split_own(df_train[cols], df_train.y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(df_train[cols], df_train.y, test_size=0.5, random_state=17*i)
    """
# Delete Outliers
if OUTLIERS:
    X_train, y_train = delete_outliers(X_train, y_train, OUTLIER_FACTOR)
    X_test = clip_outliers(X_test, OUTLIER_FACTOR)

# Create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    #'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'mse'},
    'num_leaves': 127,
    'learning_rate': 0.05,
    'bagging_fraction': 0.95, # 1
    'feature_fraction': 1, # 1
    'bagging_freq': 5,
    'max_depth': -1,
    #'max_bin': 511, # default=255
    'min_data_in_leaf': 350, # default=20
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

if SUBMISSION:
    # to submission
    df_test['y'] = y_pred
    df_test['y'].to_csv('submission40.csv', header=True)
    print(".csv submission is ready")
else:
    # metric
    rmse_accumulated += sklearn.metrics.mean_squared_error(y_test.values, y_pred) * y_test.count()
    print(str(sklearn.metrics.mean_squared_error(y_test.values, y_pred) * y_test.count()))

#print("Total: "+str(rmse_accumulated/float(N_CV+1)) + "\n")
