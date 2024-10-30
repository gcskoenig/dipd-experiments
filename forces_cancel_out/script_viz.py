import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dipd.plots import forceplot
from dipd.consts import FORCEPLOT_COLOR_DICT, PLOTS_FONT_AESTHETICS

total_color = FORCEPLOT_COLOR_DICT['loco']
seed = 42

sns.set_context('paper')
figsize = (1.2, 1.8)

savepath = 'experiments/forces_cancel_out/'

data = pd.read_csv(savepath + f'simple_{seed}.csv', index_col=0)

sns.set_context('paper')
plt.figure()
ax = forceplot(data, '', figsize=figsize, xticklabel_rotation=0, split_additive=True,
               hline_thickness=1.6, bar_width=0.8, hline_width=0.8, separator_ident_prop=0.06,
               ylabel='Normalized Scores', **PLOTS_FONT_AESTHETICS)
ax.get_legend().remove()
plt.savefig(savepath + f'simple_{seed}.pdf', bbox_inches='tight')
plt.close()


## second example

data = pd.read_csv(savepath + f'same_deps_{seed}.csv', index_col=0)

ax = forceplot(data, 'DIP Decomposition', figsize=(7,3), xticklabel_rotation=0, split_additive=True, sort_by='name')
ax = ax.get_legend().remove()
plt.savefig(savepath + f'same_deps_{seed}.pdf', bbox_inches='tight')
# plt.show()
plt.close()