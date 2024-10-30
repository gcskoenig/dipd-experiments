"""
Here we apply the DIP decomposition for LOCO on the wine dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dipd import DIP
from dipd.learners import EBM
from dipd.consts import PLOTS_FONT_AESTHETICS

from ucimlrepo import fetch_ucirepo
import random

seed = 10
random.seed(seed)
np.random.seed(seed)

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

## LOO PID decompositions for all features using 10-fold cross-validation

from sklearn.model_selection import KFold

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
result_df = pd.concat(resultsscores, axis=0)

# correct for negative pure interactions
pure_interactions = result_df['pure_interactions']
pure_interactions.iloc[np.where(result_df['pure_interactions'] <= 0)] = 0.0
result_df['pure_interactions'] = pure_interactions

# group by index and average within groups
result_df_mean = result_df.groupby(result_df.index).mean().copy()
result_df_std = result_df.groupby(result_df.index).std()
result_df_mean.to_csv(savepath + f'wine_loo_{folds}fold_{seed}_mean.csv')
result_df_std.to_csv(savepath + f'wine_loo_{folds}fold_{seed}_std.csv')

## plots for paper

from dipd.plots import forceplot

result_df_mean = pd.read_csv(savepath + f'wine_loo_{folds}fold_{seed}_mean.csv', index_col=0)

font_aesthetics = PLOTS_FONT_AESTHETICS.copy()
font_aesthetics['fontsize'] = 6.5

plt.figure()
ax = forceplot(result_df_mean.T, 'Wine Quality', figsize=(3.25, 1.5), 
               ylabel='Normalized Scores',
               separator_ident_prop=0.03,
               explain_surplus=True, xticklabel_rotation=25,
               hline_thickness=0.8, hline_width=0.8, **font_aesthetics)
ax.get_legend().remove()
plt.savefig(savepath + f'wine_loo_{folds}fold_{seed}.pdf', bbox_inches='tight')

## plots for appendix

plt.figure()
ax = forceplot(result_df_mean.T, 'Wine Quality', figsize=(6.5, 1.5), 
               ylabel='Normalized Scores', split_additive=True,
               separator_ident_prop=0.03,
               explain_surplus=True, xticklabel_rotation=25,
               hline_thickness=0.8, hline_width=0.8, **font_aesthetics)
ax.get_legend().remove()
plt.savefig(savepath + f'wine_loo_{folds}fold_{seed}_large.pdf', bbox_inches='tight')


## pairwise ablation

features = ['citric_acid', 'residual_sugar']
folds = 10

kf = KFold(n_splits=folds, shuffle=True)
splits = kf.get_n_splits(df)
X, y = df.drop(target_variable, axis=1), df[target_variable]

for feature in features:

    results = []
    i, (train_index, test_index) = next(enumerate(kf.split(df)))
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        print(f'Fold {i+1}/{folds}')
        X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
        wrk.set_split(X_train, X_test, y_train, y_test)
        ex_pairwise_of = wrk.get_all_pairwise_onefixed(feature)
        results.append(ex_pairwise_of)


    resultsscores = [result.scores for result in results]
    # append the results pd.series to a dataframe with the index indicating the fold
    result_df = pd.concat(resultsscores, axis=0)
    result_df.to_csv(savepath + f'wine_pair_{feature}_{folds}fold_{seed}.csv')

    # group by index and average within groups
    result_df_mean = result_df.groupby(result_df.index).mean().copy()
    result_df_std = result_df.groupby(result_df.index).std()
    result_df_mean.to_csv(savepath + f'wine_pair_{feature}_{folds}fold_{seed}_mean.csv')
    result_df_std.to_csv(savepath + f'wine_pair_{feature}_{folds}fold_{seed}_std.csv')


from dipd.plots import forceplot

for feature in features:
    result_df_mean = pd.read_csv(savepath + f'wine_pair_{feature}_{folds}fold_{seed}_mean.csv', index_col=0)

    plt.figure()
    ax = forceplot(result_df_mean.T, 'Pairwise DIP Decomposition',figsize=(3.25, 1.5), 
               ylabel='Normalized Scores',
               separator_ident_prop=0.03, xticklabel_rotation=25,
               hline_thickness=0.8, hline_width=0.8, **font_aesthetics)
    legend = ax.get_legend()
    ax.get_legend().remove()
    plt.savefig(savepath + f'wine_pair_{feature}_{folds}fold_{seed}.pdf', bbox_inches='tight')
    
    
    


