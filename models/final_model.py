# -*- coding: utf-8 -*-

"""Stacking of some good solutions.
IMPORTANT:
To run this model you need run before the differents models.
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv('submission40.csv') # 0.309812 (public leaderboard)
df2 = pd.read_csv('submission41.csv') # 0.305985 (public leaderboard)
df3 = pd.read_csv('submission42.csv') # 0.313587 (public leaderboard)
df4 = pd.read_csv('submission45.csv') # 0.309749 (public leaderboard)
df5 = pd.read_csv('submission47.csv') # 0.306439 (public leaderboard)

df = pd.DataFrame()

df['y'] = 0.2*df1['y'] + 0.23*df2['y'] + 0.2*df3['y'] + 0.15*df4['y'] + 0.22*df5['y']
df.to_csv('submission53.csv') # 0.301697 (public leaderboard)
