"""
In this script we apply the DIP decomposition to SAGE values on the wine dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dipd import DIP
from dipd.learners import EBM
from dipd.consts import PLOTS_FONT_AESTHETICS

from ucimlrepo import fetch_ucirepo
import random
import tqdm
import pickle

seed = 10
random.seed(seed)
np.random.seed(seed)
orders = 100

MAX_N = -1
MAX_D = -1

savepath = 'experiments/application_wine/'


## load data from remote

dataset_dict = {'bike_sharing': 275, 'wine_quality': 186, 'auto_mpg': 9, 'student_performance': 320, 'abalone': 1, 'automobile': 10, 'real_estate_valuation': 477}
dataset = fetch_ucirepo(id=dataset_dict['wine_quality'])
X_pd = dataset.data.features.iloc[:MAX_N, :MAX_D]
Y_pd = dataset.data.targets
if Y_pd is None:
    Y_pd = dataset.data.features.iloc[:, -1]
    X_pd.drop(X_pd.columns[-1], axis=1, inplace=True)
target_variable = Y_pd.columns[0]
df = pd.concat([X_pd, Y_pd.loc[:, target_variable]], axis=1)

df.dropna(inplace=True)


## computing with all variables


def sage_decomp(wrk, nr_orderings, savepath):
    
    features = wrk.X_train.columns

    # sample permutations of the features
    orderings = [np.random.permutation(features) for _ in range(nr_orderings)]

    with open(savepath + f'sage_orderings_{nr_orderings}orders_{seed}.pkl', 'wb') as f:
        pickle.dump(orderings, f)

    res = wrk.get([features[0], features[1]])
    res.iloc[:] = 0

    res_df = pd.DataFrame(columns=res.index)

    # initialize dictionary with an empty dataframe for each feature
    results = {feature: res_df.copy() for feature in features}
    for i, order in tqdm.tqdm(enumerate(orderings)):
        order = list(order)
        for j, feature in enumerate(order):
            comb = [order[:j], feature]
            res = wrk.get(comb)
            if j == 0:
                tmp = res.copy()
                res.iloc[:] = 0
                res['v2'] = tmp['v2']
            results[feature].loc[str(comb[0])] = res
        
    with open(savepath + f'sage_{nr_orderings}orders_{seed}.pkl', 'wb') as f:
        pickle.dump(results, f)
            
wrk = DIP(df, target_variable, EBM)

sage_decomp(wrk, orders, savepath)

