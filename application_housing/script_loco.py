"""
In this script, we apply the DIP decomposition to the California Housing dataset.

First, we apply a LOO DIP decomposition to a reduced version of the dataset with only three variables.
Then we apply a 10-fold DIP decomposition to the full dataset.

The script is accompanied by a _viz.py script that contains the code to visualize the results.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

from dipd import DIP
from dipd.learners import EBM

import random

seed = 42
random.seed(seed)
np.random.seed(seed)

savepath = 'experiments/application_housing/'

df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
df.dropna(inplace=True)
encoder = ce.OrdinalEncoder()
df = encoder.fit_transform(df)
target_variable = 'median_house_value'

df.columns = ['longitude', 'latitude', 'age', 'rooms',
       'bedrooms', 'population', 'households', 'income',
       'median_house_value', 'ocean']

## the three variable case

features = ['longitude', 'latitude', 'ocean', 'median_house_value']
df3 = df[features]
X = df3.drop(target_variable, axis=1)
y = df3[target_variable]


dipdwrk = DIP(df3, target_variable, EBM)
ex = dipdwrk.get_all_loo()
ex.scores.to_csv(savepath + f'3vars_loo_dip_{seed}.csv')

loco_scores = ex.scores.sum(axis=1) - ex.scores['v2']
fi_vals = loco_scores
fi_vals.to_csv(savepath + f'3vars_loco_{seed}.csv')

## computing with all variables

from sklearn.model_selection import KFold
from dipd.plots import forceplot

folds = 10

wrk = DIP(df, target_variable, EBM)

kf = KFold(n_splits=folds, shuffle=True)
splits = kf.get_n_splits(df)
X, y = df.drop(target_variable, axis=1), df[target_variable]

results = []
for i, (train_index, test_index) in enumerate(kf.split(df)):
    print(f'Fold {i+1}/{folds}')
    X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
    wrk.set_split(X_train, X_test, y_train, y_test)
    ex_loo = wrk.get_all_loo()
    scores = ex_loo.scores
    results.append(ex_loo)

resultsscores = [result.scores for result in results]
# append the results pd.series to a dataframe with the index indicating the fold
result_df = pd.concat(resultsscores, axis=0)

# group by index and average within groups
result_df_mean = result_df.groupby(result_df.index).mean().copy()
result_df_std = result_df.groupby(result_df.index).std()

result_df_mean.to_csv(savepath + f'housing_loo_{folds}fold_{seed}_mean.csv')
result_df_std.to_csv(savepath + f'housing_loo_{folds}fold_{seed}_std.csv')