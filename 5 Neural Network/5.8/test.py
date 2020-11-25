"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/11/25
- Brief: Test Self-Organizing Network with watermelon dataset 3.0alpha
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from SOM import SOM

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

SOMNet = SOM(xs.shape[1], map_shape=[8, 8])
SOMNet.learn(xs, steps=1000, batch_size=17)

# plot data (before mapping)
plt.figure()
positive_xs = xs[ys==1]
negative_xs = xs[ys==0]
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], marker='o', c='w', edgecolors='#00CED1', s=80, label='Great (positive)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], marker='s', c='w', edgecolors='#DC143C', s=80, label='Awful (negative)')
plt.legend(loc="upper right", bbox_to_anchor=(1.01, 1.16))

# map
mapped_xs = SOMNet.forward(xs)
positive_xs = mapped_xs[ys==1] + 0.5
negative_xs = mapped_xs[ys==0] + 0.5

# plot data (after mapping)
plt.figure()
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], marker='o', c='w', edgecolors='#00CED1', s=80, label='Great (positive)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], marker='s', c='w', edgecolors='#DC143C', s=80, label='Awful (negative)')
plt.axis([0, SOMNet.map_shape[0], 0, SOMNet.map_shape[1]])
ax = plt.gca()
ax.invert_yaxis()
plt.grid(linestyle='-.')
plt.legend(loc="upper right", bbox_to_anchor=(1.01, 1.16))

# distribution at each mapped position
plt.figure()
plt.axes(aspect='equal')
the_grid = GridSpec(SOMNet.map_shape[0], SOMNet.map_shape[1])
colors = ['C0', 'C1']
for idx in range(SOMNet.map_shape[0]*SOMNet.map_shape[1]):
    pos = np.array([idx//SOMNet.map_shape[1], idx%SOMNet.map_shape[1]])
    if 0 in np.sum((mapped_xs-pos)**2, axis=-1):
        plt.subplot(the_grid[pos[1], pos[0]], aspect=1)
        ys_at_this_pos = ys[np.sum((mapped_xs-pos)**2, axis=-1)==0]
        pnum = np.sum(ys_at_this_pos)
        nnum = len(ys_at_this_pos) - pnum
        plt.pie(x=[pnum, nnum], colors=colors, labels=['Great (positive)', 'Awful (negative)'], textprops={'fontsize': 0, 'color': 'w'})
        plt.text(pos[0]/100, pos[1]/100, str(pnum+nnum), color='black', fontdict={'weight': 'bold', 'size': 10}, va='center', ha='center')
plt.legend(loc="upper right", bbox_to_anchor=(4, 3))

plt.show()
