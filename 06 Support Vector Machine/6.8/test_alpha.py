"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/12/14
- Brief: Test Support Vector Regression with watermelon dataset 3.0alpha
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from SVR import SVR

# load data
data = xlrd.open_workbook('../WTMLDataSet_3.0alpha.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)

xs = dataset[1:, 1].astype(np.float64).reshape(-1, 1)
ys = dataset[1:, 2].astype(np.float64)

# fit with Support Vector Regression
svr = SVR(xs.shape[1], func='Gaussian', sigma=0.01)
svr.fit(xs, ys, C=100, epsilon=1, err=0.01, iters=100)

# plot fitted curve
x_range = np.arange(0, 1, 0.01)
preds = svr.predict(x_range)
plt.plot(x_range, preds, linestyle='--', c='skyblue', label='fit curve')
# plot source data
plt.scatter(xs, ys, c='#AC5EB8', marker='x', s=20, label='data')
plt.xlabel("x[0]: Density"); plt.ylabel("x[1]: Sugar Content")

plt.legend()
plt.show()
