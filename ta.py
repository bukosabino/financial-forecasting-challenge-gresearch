# -*- coding: utf-8 -*-

"""This functions are based on my own technical analysis library:
https://github.com/bukosabino/ta
You should check it if you need documentation of this functions.
"""

import pandas as pd
import numpy as np


"""
Volatility Indicators
"""


def bollinger_hband(close, n=20, ndev=2):
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    return pd.Series(hband, name='hband')


def bollinger_lband(close, n=20, ndev=2):
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    return pd.Series(lband, name='lband')


def bollinger_mavg(close, n=20):
    mavg = close.rolling(n).mean()
    return pd.Series(mavg, name='mavg')


def bollinger_hband_indicator(close, n=20, ndev=2):
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='bbihband')


def bollinger_lband_indicator(close, n=20, ndev=2):
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='bbilband')


def donchian_channel_hband(close, n=20):
    hband = close.rolling(n).max()
    return pd.Series(hband, name='dchband')


def donchian_channel_lband(close, n=20):
    lband = close.rolling(n).min()
    return pd.Series(lband, name='dclband')


def donchian_channel_hband_indicator(close, n=20):
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close > hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='dcihband')


def donchian_channel_lband_indicator(close, n=20):
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='dcilband')


"""
Volume Indicators
"""


def on_balance_volume(close, volume):
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    return df['OBV']


def on_balance_volume_mean(close, volume, n=10):
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    return pd.Series(df['OBV'].rolling(n).mean(), name='obv_mean')


def force_index(close, volume, n=2):
    return pd.Series(close.diff(n) * volume.diff(n), name='fi_'+str(n))


def volume_price_trend(close, volume):
    vpt = volume * ((close - close.shift(1)) / close.shift(1).astype(float))
    vpt = vpt.shift(1) + vpt
    return pd.Series(vpt, name='vpt')
