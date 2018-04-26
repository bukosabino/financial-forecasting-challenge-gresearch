# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from utils import *

# Settings
ADD_DATE_FEATURES = True
ADD_BLACK_MAGIC = True
ADD_TA_FEATURES = True
ADD_TA2_FEATURES = False

print('loading original datas...')

# Load data
df_train = pd.read_csv('../input/train.csv', index_col=0)
df_test = pd.read_csv('../input/test.csv', index_col=0)

# Merge train and test to add features
df = pd.concat([df_train, df_test], keys=['train', 'test']).sort_values(['Day'])

# Fill nan
df = fillna_bystock(df)

print('preprocessing...')

# Add "black magic" features
if ADD_BLACK_MAGIC:
    df = feature_engineering_blackmagic(df)
    print('black magic features added')

# Add date features
if ADD_DATE_FEATURES:
    df = feature_engineering_dates(df)
    print('dates features added')

# Add TA features
if ADD_TA_FEATURES:
    df = feature_engineering_ta(df)
    df = fillna_bystock(df)
    print('technical analysis features added')

# Add TA2 features
if ADD_TA2_FEATURES:
    df = feature_engineering_ta2(df)
    df = fillna_bystock(df)
    print('technical analysis 2 features added')

# Unmerge train/test
test = df.loc["test"].sort_index()
train = df.loc["train"].sort_index()

# Save datas
train.to_csv('../input/train_fe.csv', header=True)
test.to_csv('../input/test_fe.csv', header=True)
