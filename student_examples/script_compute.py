import numpy as np
import pandas as pd
import random

from dipd import DIP
from dipd.learners import EBM
from dipd.learners import LinearGAM

seed = 42

random.seed(seed)
np.random.seed(seed)


SAVEPATH = 'experiments/student_examples/'

N = 10**5

## three binary examples from the paper

# enhancement

ex_name = 'enhancement'

x = np.array([[1, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 0]])
ixs = np.random.choice(x.shape[0], N, replace=True)
x = x[ixs]
y = 4*x[:, 0] + 2*x[:, 1]

df = pd.DataFrame({'y': y, 'x1': x[:, 0], 'x2': x[:, 1]})

wrk = DIP(df, 'y', LinearGAM)

res = wrk.get(['x1', 'x2'], C=[], normalized=False)
data = pd.DataFrame({'': res})

data.to_csv(SAVEPATH + f'{ex_name}_{seed}.csv')

# redundancy

ex_name = 'redundancy'

x = np.array([[1, 1], [1, 1], [1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
ixs = np.random.choice(x.shape[0], N, replace=True)
x = x[ixs]
y = 4*x[:, 0] + 4*x[:, 1]


df = pd.DataFrame({'y': y, 'x1': x[:, 0], 'x2': x[:, 1]})

wrk = DIP(df, 'y', LinearGAM)

res = wrk.get(['x1', 'x2'], C=[], normalized=False)
data = pd.DataFrame({'': res})

data.to_csv(SAVEPATH + f'{ex_name}_{seed}.csv')


# interaction

from dipd.learners import EBM 

ex_name = 'interaction'
x = np.array([[1, 1], [1, 1], [1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
ixs = np.random.choice(x.shape[0], N, replace=True)
x = x[ixs]
x1 = x[:, 0]
x2 = x[:, 1]
y = 8 * (np.logical_or(x1, x2).astype(int)) - 1 

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

wrk = DIP(df, 'y', EBM)

res = wrk.get(['x1', 'x2'], C=[], normalized=False)
data = pd.DataFrame({'': res})

data.to_csv(SAVEPATH + f'{ex_name}_{seed}.csv')

