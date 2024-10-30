"""
The point of this example is to show that there can be totally different collaboration structures but both 
models can get the exact same attribution. In this case, even the dependence structure in the data is
exactly the same.

Maybe to get the point across it would be easier to use an example where the dependence structure is different
and therefore it is easier to see that the collaboration structure is different. In case somebody asks whether
we can reconstruct it from the dependence structure one could put this example into the appendix because it is
a little less elegant than the one where the dependence structure varies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from dipd import DIP
from dipd.learners import EBM, LinearGAM
from dipd.plots import forceplot

import random

seed = 42
random.seed(seed)
np.random.seed(seed)

savepath = 'experiments/forces_cancel_out/'

N = 10**5

## with different dependence structure

# setting 1
x = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N)
x1 = x[:, 0]
x2 = x[:, 1]
y = x1 + x2

df1 = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
wrk1 = DIP(df1, 'y', EBM)
res1 = wrk1.get(['x1', 'x2'])
res1['main_effect_cross_predictability'] = 0.0
res1['main_effect_cov'] = 0.0

# setting 2
x = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], N)
x1 = x[:, 0]
x2 = x[:, 1]
y = x1 + x2 + math.sqrt(6) * x1 * x2

df2 = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
wrk2 = DIP(df2, 'y', EBM)
res2 = wrk2.get(['x1', 'x2'])

data = pd.DataFrame({'DGP1': res1, 'DGP2': res2})
data.to_csv(savepath + f'simple_{seed}.csv')


## with the same dependence structure

N = 10**6


x = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], N)
x1 = x[:, 0]
x2 = x[:, 1]

# get the data
y2 = -1.3 * x1 - 4.7 * x2 + 3.6 * x1**2 - 3 * x2**2 + 4.7 * x1 * x2
y1 = -4.3 * x1 - 0.9 * x2 - 3.9 * x1**2 + 3 * x2**2
y3 = 10.9 * x1 + 2.4 * x2 - 5.1 * x1**2 - 5.3 * x2**2 + 11.3 * x1 * x2

df1 = pd.DataFrame({'y': y1, 'x1': x1, 'x2': x2})
df2 = pd.DataFrame({'y': y2, 'x1': x1, 'x2': x2})
df3 = pd.DataFrame({'y': y3, 'x1': x1, 'x2': x2})

# compute setting 1
print('DGP 1')
wrk = DIP(df1, 'y', EBM)
res1 = wrk.get(['x1', 'x2'])

print('DGP 2')
wrk = DIP(df2, 'y', EBM)
res2 = wrk.get(['x1', 'x2'])

print('DGP 3')
wrk = DIP(df3, 'y', EBM)
res3 = wrk.get(['x1', 'x2'])

data = pd.DataFrame({'DGP 1': res1, 'DGP 2': res2, 'DGP 3': res3})

data.to_csv(savepath + f'same_deps_{seed}.csv')
