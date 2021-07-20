"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2021/06/17
- Brief: implement Bagging based on pruning-free decision tree with watermelon dataset 3.0α
"""

import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from DecisionTree_v4 import DecisionTree
from Bagging import Bagging

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
isdiscs = np.array([0, 0])
labels = ["坏瓜", "好瓜"]

print("Test Accuracy:")

# use decision tree with depth 2 as the base learner
decisionTree = DecisionTree(xs, ys, xs, ys, attributes, isdiscs, labels)
decisionTree.setup(MaxDepth=2, partIndex='InformationGain', prepruning=False)

plt.figure()
base_nums = [1, 10, 200]
for it in range(len(base_nums)):
    bagging = Bagging(decisionTree, base_nums[it])
    bagging.fit(xs, ys)
    print("- Decision Tree (max depth -- 2) x %d\t:    %.2f"%(base_nums[it], bagging.test(xs, ys)))

    plt.subplot(1, 3, it+1)
    # plot data
    positive_xs = xs[ys==1]
    negative_xs = xs[ys==0]
    plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', s=60, label='Great (positive)')
    plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', s=60, label='Awful (negative)')
    plt.xlabel("x[0]: Density")
    plt.ylabel("x[1]: Sugar Content")

    # show partition
    x0, x1 = 100, 100
    x0, x1 = np.meshgrid(np.linspace(0, 1, x0), np.linspace(0, 1, x1))
    datum = np.concatenate([x0.reshape(10000, 1), x1.reshape(10000, 1)], axis=-1)
    preds = np.array([bagging.classify(x) for x in datum]).reshape(x0.shape)
    plt.contour(x0, x1, preds, [0.5], linewidths=2, colors='m', label='Partition')

    plt.title("Decision Tree (max depth -- 2) x %d"%base_nums[it])
    plt.legend()

plt.show()