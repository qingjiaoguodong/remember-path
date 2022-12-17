from remember_path import RememberPath as p

a, b = p(0), p(0)
result = 1 + a * (a + b + b * b + 0)
assert result.i == 1

a.i = 1 # b * b is cached
assert result.i == 1 + 1 * (1 + 0 + 0 * 0 + 0)

b.i = 2
assert result.i == 1 + 1 * (1 + 2 + 2 * 2 + 0)

a.i, b.i = 1, 3
a += 1
assert result.i == 1 + 2 * (2 + 3 + 3 * 3 + 0)

import numpy as np
import pandas as pd

npe = p(np)
a = p(np.zeros(5))
b = p(np.zeros(5))
c = np.zeros(5)
r = npe.outer(a, a + b) + c
print(r.i)
a.i = np.ones(5)
print(r.i)

df = p(pd.DataFrame())
dfp = df + 100
print(df.i)
df.index = [0, 1]
df['r'] = [2, 3]
print(df.i)
df.iloc[1] = 9
print(dfp.i)
df += 1000
df.iloc[0] -= 5000
print(dfp.i)

'''
[[0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]]
[[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]
Empty DataFrame
Columns: []
Index: []
    r
0  2
1  3
      r
0  102
1  109
      r
0 -3898
1  1109
'''

