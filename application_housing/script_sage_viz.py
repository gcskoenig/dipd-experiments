import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import re
import pickle

from dipd.plots import forceplot
from dipd.learners import LinearGAM
from dipd.consts import PLOTS_FONT_AESTHETICS

savepath = 'experiments/application_housing/'

norders = 100
seed = 42
# files = os.listdir(savepath)
# files = [f for f in files if re.match(r'sage_10orders_.*\.pkl', f)]
# files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(0)), reverse=True)
filename = f'sage_{norders}orders_{seed}.pkl'    
    
with open(savepath + filename, 'rb') as f:
    results_loaded = pickle.load(f)

fs = list(results_loaded.keys())
columns = results_loaded[fs[0]].columns

scores = pd.DataFrame(columns=columns, index=fs)
scores_std = pd.DataFrame(columns=columns, index=fs)
for feature in results_loaded.keys():
    res_mean = results_loaded[feature].mean(axis=0)
    res_std = results_loaded[feature].std(axis=0)
    scores.loc[feature, :] = res_mean
    scores_std.loc[feature, :] = res_std
    
scores_plt = scores.copy()
scores_plt['v1'] = scores['v2'].to_numpy()
    
font_aesthetics = PLOTS_FONT_AESTHETICS.copy()
font_aesthetics['fontsize'] = 6.5
    
plt.figure()
ax = forceplot(scores_plt.T, 'California Housing', figsize=(3.25, 1.5), ylabel='Normalized Scores',
               separator_ident_prop=0.03, xticklabel_rotation=25, hline_thickness=0.8, hline_width=0.8, explain_surplus=True, **font_aesthetics)
ax.get_legend().remove()
plt.savefig(savepath + f'sage_{norders}orders_{seed}.pdf', bbox_inches='tight')  