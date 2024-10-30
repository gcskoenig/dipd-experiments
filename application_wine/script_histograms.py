"""
Here we further analyze the relationship between residual sugar and density (and their roles for the target).
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dipd.consts import PLOTS_FONT_AESTHETICS

from ucimlrepo import fetch_ucirepo
import random

seed = 10
random.seed(seed)
np.random.seed(seed)

MAX_N = -1
MAX_D = -1

savepath = 'experiments/application_wine/'

##

font_aesthetics = PLOTS_FONT_AESTHETICS.copy()
font_aesthetics['fontsize'] = 6.5

plt.rcParams.update({
    'font.size': font_aesthetics['fontsize'],  # Set the global font size
    'font.family': font_aesthetics['fontname'],  # Set the global font family
    'axes.titlesize': font_aesthetics['fontsize'],  # Set the font size for axes titles
    'axes.labelsize': font_aesthetics['fontsize'],  # Set the font size for axes labels
    'xtick.labelsize': font_aesthetics['fontsize'],  # Set the font size for x-tick labels
    'ytick.labelsize': font_aesthetics['fontsize'],  # Set the font size for y-tick labels
    'legend.fontsize': font_aesthetics['fontsize'],  # Set the font size for legend
    'figure.titlesize': font_aesthetics['fontsize']  # Set the font size for figure title
})


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

## get residual sugar and density quartiles

df['residual_sugar_quantile'] = pd.qcut(df['residual_sugar'], q=20, labels=False)
df['density_quantile'] = pd.qcut(df['density'], q=20, labels=False)
df['residual_sugar_quartile'] = pd.qcut(df['residual_sugar'], q=6, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'])
df['density_quartile'] = pd.qcut(df['density'], q=6, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'])



## pairwise histplots for residual_sugar_quantile with the target_variable and density_quantile with the target_variable

pmax = 0.95

plt.figure(figsize=(1.1, 1.1))
sns.histplot(data=df, x='residual_sugar_quantile', y=target_variable, 
             discrete=(True, True), cmap="light:#03012d", pmax=pmax, cbar=False)
plt.title('')
plt.ylabel('Quality')
plt.xlabel('Sugar Quantile')

# Remove axes, ticks, and ticklabels
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left=False, bottom=False)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig(savepath + 'hist_pair_sugar_quality.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(1.1, 1.1))
sns.histplot(data=df, x='density_quantile', y=target_variable, 
             discrete=(True, True), cmap="light:#03012d", pmax=pmax, cbar=False)
plt.title('')
plt.ylabel('Quality')
plt.xlabel('Density Quantile')

# Remove axes, ticks, and ticklabels
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left=False, bottom=False)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig(savepath + 'hist_pair_density_quality.pdf', bbox_inches='tight')
plt.close()


## plot of their joint relationship with the target

figsize = (5.5, 1.1)

# Load the planets dataset and initialize the figure
g = sns.FacetGrid(data=df, col='residual_sugar_quartile')
g.map_dataframe(
    sns.histplot, x='density_quantile', y=target_variable, 
    discrete=(True, True),
    cmap="light:#03012d", pmax=pmax, cbar=False
)

# Adjust the layout
g.set_titles(col_template="{col_name}", fontsize=font_aesthetics['fontsize'])
g.set_axis_labels("Density", "Quality")
g.fig.set_size_inches(*figsize)

for ax in g.axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig(savepath + 'hists_density_quality_by_sugar.pdf', bbox_inches='tight')
plt.close()


# Load the planets dataset and initialize the figure
g = sns.FacetGrid(data=df, col='density_quartile')
g.map_dataframe(
    sns.histplot, x='residual_sugar_quantile', y=target_variable, 
    discrete=(True, True),
    cmap="light:#03012d", pmax=pmax, cbar=False
)

# Adjust the layout
g.set_titles(col_template="{col_name}")
g.set_axis_labels("residual_sugar", "quality")
g.fig.set_size_inches(*figsize)

for ax in g.axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig(savepath + 'hists_sugar_quality_by_density.pdf', bbox_inches='tight')
plt.close()


## citric acidity and volatile acidity

## get residual sugar and density quartiles

df['volatile_acidity_quantile'] = pd.qcut(df['volatile_acidity'], q=20, labels=False)
df['citric_acid_quantile'] = pd.qcut(df['citric_acid'], q=20, labels=False)
df['volatile_acidity_quartile'] = pd.qcut(df['volatile_acidity'], q=6, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'])
df['citric_acid_quartile'] = pd.qcut(df['citric_acid'], q=6, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'])



## pairwise histplots for residual_sugar_quantile with the target_variable and density_quantile with the target_variable

pmax = 0.85

plt.figure(figsize=(1.1, 1.1))
sns.histplot(data=df, x='volatile_acidity_quantile', y=target_variable, 
             discrete=(True, True), cmap="light:#03012d", pmax=pmax, cbar=False)
plt.title('')
plt.ylabel('Quality')
plt.xlabel('Volatile Acidity')

# Remove axes, ticks, and ticklabels
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left=False, bottom=False)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig(savepath + 'hist_pair_volatile_quality.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(1.1, 1.1))
sns.histplot(data=df, x='citric_acid_quantile', y=target_variable, 
             discrete=(True, True), cmap="light:#03012d", pmax=pmax, cbar=False)
plt.title('')
plt.ylabel('Quality')
plt.xlabel('Citric Quantile')

# Remove axes, ticks, and ticklabels
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left=False, bottom=False)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig(savepath + 'hist_pair_citric_quality.pdf', bbox_inches='tight')
plt.close()


## plot of their joint relationship with the target

figsize = (5.5, 1.1)

# Load the planets dataset and initialize the figure
g = sns.FacetGrid(data=df, col='volatile_acidity_quartile')
g.map_dataframe(
    sns.histplot, x='citric_acid_quantile', y=target_variable, 
    discrete=(True, True),
    cmap="light:#03012d", pmax=pmax, cbar=False
)

# Adjust the layout
g.set_titles(col_template="{col_name}", fontsize=font_aesthetics['fontsize'])
g.set_axis_labels("Citric", "Quality")
g.fig.set_size_inches(*figsize)

for ax in g.axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig(savepath + 'hists_citric_quality_by_volatile.pdf', bbox_inches='tight')
plt.close()


# Load the planets dataset and initialize the figure
g = sns.FacetGrid(data=df, col='citric_acid_quartile')
g.map_dataframe(
    sns.histplot, x='volatile_acidity_quantile', y=target_variable, 
    discrete=(True, True),
    cmap="light:#03012d", pmax=pmax, cbar=False
)

# Adjust the layout
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Volatile Acidity", "quality")
g.fig.set_size_inches(*figsize)

for ax in g.axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig(savepath + 'hists_volatile_quality_by_citric.pdf', bbox_inches='tight')
plt.close()

## compute correlations

columns = ['residual_sugar', 'density', 'quality']
conditioning_vars = ['residual_sugar_quartile', 'density_quartile']

fs = columns + conditioning_vars

corr_density = df[columns + ['residual_sugar_quartile']].groupby('residual_sugar_quartile').corr().loc[(slice(None), 'quality'), ['density']].reset_index()
corr_sugar = df[columns + ['density_quartile']].groupby('density_quartile').corr().loc[(slice(None), 'quality'), ['residual_sugar']].reset_index()

agg_sugar = corr_sugar['residual_sugar'].mean()
agg_density = corr_density['density'].mean()

plain_corr = df[columns].corr()['quality'].loc[['residual_sugar', 'density']]

print('Plain correlations with quality: sugar {} and density {}'.format(plain_corr['residual_sugar'], plain_corr['density']))
print('Average absolute correlation with quality: sugar {} and density {}'.format(agg_sugar, agg_density))

corrs = df[fs].groupby(conditioning_vars).corr().loc[(slice(None), 'quality'), :].reset_index()