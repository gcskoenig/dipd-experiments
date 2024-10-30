import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dipd.plots import forceplot
from dipd.consts import PLOTS_FONT_AESTHETICS

seed = 42
savepath = 'experiments/application_housing/'

fi_vals = pd.read_csv(savepath + f'3vars_loco_{seed}.csv', index_col=0)

plt.figure(figsize=(5, 5))
ax = sns.barplot(fi_vals)
sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
plt.savefig(savepath + f'3vars_loco_{seed}pdf')
plt.close()

scores = pd.read_csv(savepath + f'3vars_loo_dip_{seed}.csv', index_col=0)
ax = forceplot(scores.T, '', figsize=(5, 5))
ax.get_legend().remove()
plt.savefig(savepath + f'3vars_dip_{seed}.pdf')


## all in one figure

font_aesthetics = PLOTS_FONT_AESTHETICS.copy()
font_aesthetics['fontsize'] = 6.5

sns.set_context('paper')
plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(3.25, 1.6))
axs[1] = forceplot(scores.T, '', ax=axs[1], xticklabel_rotation=0, 
                   hline_thickness=1, hline_width=0.8, explain_surplus=True, 
                   **font_aesthetics)
axs[1].get_legend().remove()
axs[1].set_title('DIP Decomposition', **font_aesthetics)
axs[1].set_ylabel('')
axs[0] = sns.barplot(fi_vals.T, ax=axs[0], color='black')
axs[0].set_title('Feature Importance', **font_aesthetics)
axs[0].set_ylim(axs[1].get_ylim())
axs[0].axhline(0, color='black', linewidth=1)
axs[0].tick_params(axis='both', which='major',
                   labelsize=font_aesthetics['fontsize'],
                   labelfontfamily=font_aesthetics['fontname'])  
axs[0].set_ylabel('Normalized Scores', **font_aesthetics)
sns.despine(top=True, right=True, left=True, bottom=True, ax=axs[0])
plt.tight_layout()
plt.savefig(savepath + f'3vars_all_{seed}.pdf')


## now the 10 fold case

folds = 10
result_df_mean = pd.read_csv(savepath + f'housing_loo_{folds}fold_{seed}_mean.csv', index_col=0)

sns.set_context('paper')
plt.figure()
ax = forceplot(result_df_mean.T, 'California Housing', figsize=(3.25, 1.5), ylabel='Normalized Scores',
               separator_ident_prop=0.03,
               explain_surplus=True, xticklabel_rotation=25,
               hline_thickness=0.8, hline_width=0.8, **font_aesthetics)
ax.get_legend().remove()
# ax.set_position([0.1, 0.3, 0.8, 0.6])
# ax.set_title('Decomposition of LOCO Feature Importance', fontsize=16)  # Set title font size
plt.savefig(savepath + f'housing_loo_{folds}fold_{seed}.pdf', bbox_inches='tight')


## plot for appendix

sns.set_context('paper')
plt.figure()
ax = forceplot(result_df_mean.T, 'California Housing', figsize=(6.5, 1.5), ylabel='Normalized Scores',
               separator_ident_prop=0.03, split_additive=True,
               explain_surplus=True, xticklabel_rotation=25,
               hline_thickness=0.8, hline_width=0.8, **font_aesthetics)
ax.get_legend().remove()
# ax.set_position([0.1, 0.3, 0.8, 0.6])
# ax.set_title('Decomposition of LOCO Feature Importance', fontsize=16)  # Set title font size
plt.savefig(savepath + f'housing_loo_{folds}fold_{seed}_large.pdf', bbox_inches='tight')