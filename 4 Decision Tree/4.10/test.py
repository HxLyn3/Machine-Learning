"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.11.07
- Brief: Deal with watermelon dataset 3.0α with multi-variate decision tree
"""

import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from MultiVarDecisionTree import MultiVarDecisionTree

# load data
data = xlrd.open_workbook('../WTMLDataSet_3.0alpha.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)

xs = dataset[1:, 1:-1].astype(np.float64)
ys = (dataset[1:, -1]=='是').astype(np.int32)
attributes = dataset[0][1:-1]
labels = ["坏瓜", "好瓜"]

# plot data
positive_xs = xs[ys==1]
negative_xs = xs[ys==0]
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', s=60, label='Great (positive)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', s=60, label='Awful (negative)')

# generate multi-variate decision tree
MVDT = MultiVarDecisionTree(xs, ys, attributes, labels)
MVDT.buildTree()
MVDT.visualize()

# partition
x0, x1 = 100, 100
x0, x1 = np.meshgrid(np.linspace(0, 1, x0), np.linspace(0, 1, x1))
xs = np.concatenate([x0.reshape(10000, 1), x1.reshape(10000, 1)], axis=-1)
preds = np.array([MVDT.classify(x) for x in xs]).reshape(x0.shape)
plt.contour(x0, x1, preds, [0.5], linewidths=2, colors='m', label='Partition')

plt.xlabel("x[0]: Density")
plt.ylabel("x[1]: Sugar Content")
plt.legend()
plt.show()