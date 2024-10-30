import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dipd.plots import forceplot
from dipd.consts import PLOTS_FONT_AESTHETICS

sns.set_context('paper')

seed = 42
SAVEPATH = 'experiments/student_examples/'

figsize = (0.7, 2.3)
ylabel = 'Scores'
thickness_bars = 1

examples = ['enhancement', 'redundancy', 'interaction']

for example in examples:
    data = pd.read_csv(SAVEPATH + f'{example}_{seed}.csv', index_col=0)
    ax = forceplot(data, '', split_additive=True, figsize=figsize, xticks=False, ylabel=ylabel, hline_thickness=thickness_bars, **PLOTS_FONT_AESTHETICS)
    ax.get_legend().remove()
    ax.set_title('')
    plt.savefig(SAVEPATH + f'{example}_{seed}.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()
