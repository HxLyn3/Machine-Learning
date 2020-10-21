"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.10.19
- Brief: Test Decision Tree with no-pruning, pre-pruning and post-pruning 
on watermelon dataset 2.0.
"""
import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
from DecisionTree_v2 import DecisionTree

# load data
data = xlrd.open_workbook('../WTMLDataSet_2.0.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)
dataset = dataset[:, [0,5,1,2,3,4,6,7]]     # adjust attributes priority

xs = dataset[1:, 1:-1]
ys = (dataset[1:, -1]=='否').astype(np.int32)
attributes = dataset[0][1:-1]
isdiscs = np.array([not bool(v.match(val)) for val in xs[0]])
labels = ["好瓜", "坏瓜"]

train_indices = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
test_indices = [3, 4, 7, 8, 10, 11, 12]

train_xs = xs[train_indices]
train_ys = ys[train_indices]
test_xs = xs[test_indices]
test_ys = ys[test_indices]

print("Accuracy:")

# decision tree
decisionTree = DecisionTree()
# non pruning
decisionTree.buildTree(train_xs, train_ys, test_xs, test_ys, \
    attributes, isdiscs, labels, partIndex='GiniIndex', prepruning=False)
print("No-Pruning    --  %.3f"%decisionTree.test(test_xs, test_ys))
decisionTree.visualize(graph_name="No-Pruning")
# post pruning
decisionTree.post_pruning()
print("Post-Pruning  --  %.3f"%decisionTree.test(test_xs, test_ys))
decisionTree.visualize(graph_name="Post-Pruning")
# pre pruning
decisionTree.buildTree(train_xs, train_ys, test_xs, test_ys, \
    attributes, isdiscs, labels, partIndex='GiniIndex', prepruning=True)
print("Pre-Pruning   --  %.3f"%decisionTree.test(test_xs, test_ys))
decisionTree.visualize(graph_name="Pre-Pruning")