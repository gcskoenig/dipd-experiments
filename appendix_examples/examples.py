import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

savepath = 'experiments/appendix_examples/figures/'

sns.set_context('paper')
sns.set_style('white')

# DGP 1
def dgp1(n, seed=0):
    y = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    eps1 = np.random.normal(0, 0.2, size=n)
    x1 = y + x2 + eps1
    
    return pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

# DGP 2
def dgp2(n, seed=0):
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    epsy = np.random.normal(0, 0.2, size=n)
    y = x1*x2 + epsy
    
    return pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

# DGP 3
def dgp3(n, seed=0):
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    epsy = np.random.normal(0, 0.2, size=n)
    y = x1 + x2 + epsy
    
    return pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

N = 10**4

df1 = dgp1(N)
df2 = dgp2(N)
df3 = dgp3(N)


from sklearn.linear_model import LinearRegression
from dipd.learners import LinearGAM


def do_df(df, fname, fvals=None, n=1):
    if fvals is None:
        fvals = df[fname].quantile(np.arange(0, 1.1, 0.1))
        fvals = pd.DataFrame({fname: fvals})
    X = df.drop('y', axis=1)
    X_ = X.sample(n=n)
    Xs = []
    for fval in fvals[fname]:
        tmp = X_.copy()
        tmp[fname] = fval
        Xs.append(tmp)
    Xs = pd.concat(Xs)
    return Xs

def get_preds(df, fname, fvals=None, n_sample=100):
    ggam = LinearGAM(interactions=1)
    ggam.fit(df.drop('y', axis=1), df['y'])
    unv = LinearRegression()
    unv.fit(df[[fname]], df['y'])
    
    Xs = do_df(df, fname, fvals=fvals, n=n_sample)
    res = Xs.copy()
    
    res['pred_univ'] = unv.predict(Xs[[fname]])
    res['pred_component'] = ggam.predict_component(Xs, (fname,))
    res['pred'] = ggam.predict(Xs)
    return res

def plots(preds, fname, ax):
    legend_created = False
    for ix in preds.index.unique():
        kwargs = {'label': 'GGAM prediction'} if not legend_created else {}
        legend_created = True
        tmp = preds.loc[ix, :]
        sns.lineplot(data=tmp, x=fname, y='pred', color='gray', alpha=0.1, ax=ax, **kwargs)
    sns.lineplot(data=preds, x=fname, y='pred_univ', label='cPDP', ax=ax, linewidth=5)
    sns.lineplot(data=preds, x=fname, y='pred_component', label='GGAM component', ax=ax, linewidth=2)
    ax.get_legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

figsize = (7, 3)
fname = 'x2'

fig, axes = plt.subplots(1, 3, figsize=figsize)  # Create a figure with 3 subplots
dgps = {'DGP 1': dgp1, 'DGP 2': dgp2, 'DGP 3': dgp3}
for ax, (dgp_name, dgp) in zip(axes, dgps.items()):
    df = dgp(N)
    preds = get_preds(df, fname, n_sample=100)
    plots(preds, fname, ax)
    ax.set_title(dgp_name)
    
# # Get handles and labels from the first subplot
# handles, labels = axes[0].get_legend_handles_labels()
# # Add a single legend to the figure
# fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend().remove()
plt.tight_layout()
plt.savefig(savepath + f'cpdp-dip.pdf', bbox_inches='tight')
plt.close()

from dipd import DIP
from dipd.plots import forceplot


fname_scores = []
dgps = {'DGP 1': df1, 'DGP 2': df2, 'DGP 3': df3}
for ax, (dgp_name, df) in zip(axes, dgps.items()):
    wrk = DIP(df, 'y', LinearGAM)
    ex = wrk.get_all_loo()
    scores = ex.scores.loc[fname, :]
    scores.name = dgp_name
    fname_scores.append(scores)
fname_scores = pd.concat(fname_scores, axis=1)

forceplot(fname_scores, f'DIP Decompositions for {fname}', split_additive=True, 
          sort_by='name', figsize=(4, 3))
plt.legend(loc='center right', bbox_to_anchor=(1, 0.5)).remove()
plt.savefig(savepath + 'dip-scores.pdf', bbox_inches='tight')
plt.close()