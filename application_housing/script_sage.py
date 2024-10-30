"""
The point of this example is to demonstrates that a feature attribution method like
conditional feature importance or SAGE does not really explain what's going on in
the model in terms of collaborations. It should be something intuitive and easy to
understand where also collaborations are visible.

We could for example take the California housing dataset and show that the SAGE values
do not really explain what's going on in the data.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

from dipd import DIP
from dipd.learners import EBM

import pickle
import random

seed = 42
nr_orders = 100
np.random.seed(seed)
random.seed(seed)

savepath = 'experiments/application_housing/'

df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
df.dropna(inplace=True)
encoder = ce.OrdinalEncoder()
df = encoder.fit_transform(df)
target_variable = 'median_house_value'

df.columns = ['longitude', 'latitude', 'age', 'rooms',
       'bedrooms', 'population', 'households', 'income',
       'median_house_value', 'ocean']

## computing with all variables

import tqdm

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

sage_decomp(wrk, nr_orders, savepath)

