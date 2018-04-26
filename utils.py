# -*- coding: utf-8 -*-
""" This is a collection of utils function to preprocessing, feature
engineering, split datasets, normalization, etc.
"""


import pandas as np
import numpy as np

from ta import *
from ta2 import *


def feature_engineering_dates(df):
    import pandas as pd
    from datetime import timedelta
    df['Date'] = pd.to_datetime("'2015-10-01'") # I guess this date is right :P
    df['Date'] = df['Date'] + df['Day'].map(timedelta) - timedelta(days=1)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.weekofyear
    df['Weekday'] = df['Date'].dt.weekday
    df['DayMonth'] = df['Date'].dt.day
    return df


def feature_engineering_ta(df):
    df_grouped = df.groupby(['Stock'])
    cols = ['x1', 'x2', 'x3E']
    v = 'x0'
    for c in cols:

        # bollinger bands
        df[c+'_lb'] = df_grouped[c].apply(lambda x: bollinger_lband(x))
        df[c+'_mb'] = df_grouped[c].apply(lambda x: bollinger_mavg(x))
        df[c+'_hb'] = df_grouped[c].apply(lambda x: bollinger_hband(x))
        df[c+'_hband'] = df_grouped[c].apply(lambda x: bollinger_hband_indicator(x))
        df[c+'_lband'] = df_grouped[c].apply(lambda x: bollinger_lband_indicator(x))

        # donchian
        df[c+'_dclband'] = df_grouped[c].apply(lambda x: donchian_channel_lband(x))
        df[c+'_dchband'] = df_grouped[c].apply(lambda x: donchian_channel_hband(x))
        df[c+'_dcihband'] = df_grouped[c].apply(lambda x: donchian_channel_hband_indicator(x))
        df[c+'_dcilband'] = df_grouped[c].apply(lambda x: donchian_channel_lband_indicator(x))

        # volume
        df[c+'_obv'] = df_grouped.apply(lambda x: on_balance_volume(x[c], x[v]))
        df[c+'_obv_mean'] = df_grouped.apply(lambda x: on_balance_volume_mean(x[c], x[v]))
        df[c+'_fi'] = df_grouped.apply(lambda x: force_index(x[c], x[v]))

    return df


def feature_engineering_ta2(df):
    c = 'x4'
    df_grouped = df.groupby(['Stock'])[c]

    df[c+'_hband'] = df_grouped.apply(lambda x: bollinger_hband_indicator(x, 10, 2))
    df[c+'_lband'] = df_grouped.apply(lambda x: bollinger_lband_indicator(x, 10, 2))

    df[c+'_dclband'] = df_grouped.apply(lambda x: donchian_channel_lband_indicator(x, 10))
    df[c+'_dchband'] = df_grouped.apply(lambda x: donchian_channel_hband_indicator(x, 10))

    df[c+'_macd'] = df_grouped.apply(lambda x: macd(x, 6, 13))
    df[c+'_macd_signal'] = df_grouped.apply(lambda x: macd_signal(x, 6, 13, 5))
    df[c+'_macd_diff'] = df_grouped.apply(lambda x: macd_diff(x, 6, 13, 5))

    df[c+'_trix'] = df_grouped.apply(lambda x: trix(x, 8))
    df[c+'_dpo'] = df_grouped.apply(lambda x: dpo(x, 10))

    df[c+'_dr'] = df_grouped.apply(lambda x: daily_return(x))
    df[c+'_cr'] = df_grouped.apply(lambda x: cumulative_return(x))

    df[c+'_rsi'] = df_grouped.apply(lambda x: rsi(x, 7))
    df[c+'_tsi'] = df_grouped.apply(lambda x: tsi(x, 13, 7))

    return df


# add useful features to market X
def feature_engineering_blackmagic(df, n=5):
    df_grouped = df.groupby(['Stock'])
    df_inverse_grouped = df[::-1].groupby(['Stock'])
    cols = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
    for c in cols:
        # rolling
        for h in [3, 5, 10, 15, 20]:
            df[c+'_rolling_mean_'+str(h)] = df_grouped[c].apply(lambda x: x.rolling(h, min_periods=0).mean())
            df[c+'_inverse_rolling_mean_'+str(h)] = df_inverse_grouped[c].apply(lambda x: x.rolling(h, min_periods=0).mean())[::-1]
        # diffs
        df[c+'_diff_1'] = df_grouped[c].apply(lambda x: x.diff().fillna(method="backfill").fillna(0))
        df[c+'_diff_2'] = df_grouped[c].apply(lambda x: x.diff(2).fillna(method="backfill").fillna(0))
        df[c+'_diff_3'] = df_grouped[c].apply(lambda x: x.diff(3).fillna(method="backfill").fillna(0))
        # cumsum
        df[c+'_cumsum'] = df_grouped[c].apply(lambda x: x.cumsum())
        # shift columns
        df[c+'_shift'] = df_grouped[c].apply(lambda x: x.shift(-1).fillna(0))
        # volatility
        df[c+'_volatility'] = 0
        df.loc[abs(df[c]-df[c].mean()) > (n*df[c].std()), c+'_volatility'] = 1
    return df


# Split dataset (train/test and X/y) alternating rows test and train
def train_test_split_own(X, y):
    trainfilter = [False if i%2 == 0 else True for i in range(X.shape[0])]
    testfilter = [True if i%2 == 0 else False for i in range(X.shape[0])]
    train = [False if i%2 == 0 else True for i in range(y.shape[0])]
    test = [True if i%2 == 0 else False for i in range(y.shape[0])]
    return X[trainfilter], X[testfilter], y[train], y[test]


# Split dataset (only train/test) alternating rows test and train
def train_test_split_own2(dframe):
    trainfilter = [False if i%2 == 0 else True for i in range(dframe.shape[0])]
    testfilter = [True if i%2 == 0 else False for i in range(dframe.shape[0])]
    return dframe[trainfilter], dframe[testfilter]


# Fill missing values in a dataframe
def fillna_bystock(df):
    for c in df.columns:
        df[c] = df.groupby(['Stock'])[c].apply(lambda x: x.fillna(method="ffill").fillna(method="backfill").fillna(0))
    return df


# XGBoost results
def print_xgboost_metric(model):
    import operator

    print("\n \n \n ********** WEIGHT ************")
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    for i in importance:
        print(i)

    print("\n \n \n ********** GAIN ************")
    importance = model.get_score(fmap='', importance_type='gain')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    for i in importance:
        print(i)


# ExtraTrees and RandomForest results
def print_ensemble_metric(model, columns):
    for idx, value in enumerate(columns):
        print(str(value) + ': ' + str(model.feature_importances_[idx]))


# Delete outliers values
def delete_outliers(X_train, y_train, n):
    import numpy as np
    cols = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
    # Delete ouliers in train
    X_train[cols] = X_train[cols][np.abs(X_train[cols]-X_train[cols].mean())<=(n*X_train[cols].std())] #keep only the ones that are within +n to -n standard deviations in the column 'Data'.
    X_train['y'] = y_train
    X_train.dropna(inplace=True)
    y_train = X_train['y']
    return X_train, y_train


# Clip outliers values
def clip_outliers(X_test, n):
    cols = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
    mean_values = X_test[cols].mean(axis=0)
    std_values = X_test[cols].std(axis=0)
    X_test[cols] = X_test[cols].clip(mean_values-std_values*n, mean_values+std_values*n, axis=1)
    return X_test


# Delete outliers values
def delete_outliers_y(X_train, y_train, n):
    import numpy as np
    cols = ['y']
    # Delete ouliers in train
    X_train[cols] = X_train[cols][np.abs(X_train[cols]-X_train[cols].mean())<=(n*X_train[cols].std())] #keep only the ones that are within +n to -n standard deviations in the column 'Data'.
    X_train['y'] = y_train
    X_train.dropna(inplace=True)
    y_train = X_train['y']
    return X_train, y_train


# Clipea outliers values
def clip_outliers_y(X_test, n):
    cols = ['y']
    mean_values = X_test[cols].mean(axis=0)
    std_values = X_test[cols].std(axis=0)
    X_test[cols] = X_test[cols].clip(mean_values-std_values*n, mean_values+std_values*n, axis=1)
    return X_test


def normalize_column(column, method='max'):
    """Normalize columns. Please note that it doesn't modify the original dataset, it just returns a new array that you can use to modify the original dataset or create a new one.
    """
    if method == 'max':
        return column/column.max()
    elif method == 'diff':
        return (column-column.min())/(column.max()-column.min())
    elif method == 'std':
        return (column-column.mean())/column.std()
    else:
        return column
    