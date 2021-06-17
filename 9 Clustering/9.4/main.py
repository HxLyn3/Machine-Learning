"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2021/06/17
- Brief: test K-means with watermelon dataset 4.0
"""

import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
import scipy.spatial as spt
import matplotlib.pyplot as plt
from Kmeans import Kmeans

# load data
data = xlrd.open_workbook('../WTMLDataSet_4.0.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)
xs = dataset[1:, 1:].astype(np.float64)

plt.figure()
# value of k for K-means
ks = [3, 4, 5]
for it in range(len(ks)):
    kmeans = Kmeans(ks[it])
    clusters, means = kmeans.cluster(xs)

    plt.subplot(1, 3, it+1)
    for idx in range(len(clusters)):
        plt.scatter(clusters[idx][:, 0], clusters[idx][:, 1], s=60, label='cluster %d'%idx)
        hull = spt.ConvexHull(clusters[idx], incremental=False)
        for sim in hull.simplices: plt.plot(clusters[idx][sim, 0], clusters[idx][sim, 1], 'm', linewidth=0.5)

    plt.scatter(means[:, 0], means[:, 1], s=40, marker='+', c='m')
    plt.xlabel("x[0]: Density")
    plt.ylabel("x[1]: Sugar Content")
    plt.title("k=%d"%ks[it])
    plt.axis([0, 1, 0, 1])
    plt.legend()

plt.show()