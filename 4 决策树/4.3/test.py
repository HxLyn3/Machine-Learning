"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.10.12
- Brief: Deal with watermelon dataset 3.0 with decision tree
"""
import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
from DecisionTree import DecisionTree

# load data
data = xlrd.open_workbook('../WTMLDataSet_3.0.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)

xs = dataset[1:, 1:-1]
ys = (dataset[1:, -1]=='是').astype(np.int32)
attributes = dataset[0][1:-1]
isdiscs = np.array([not bool(v.match(val)) for val in xs[0]])
labels = ["坏瓜", "好瓜"]

# generate decision tree
decisionTree = DecisionTree()
decisionTree.buildTree(xs, ys, attributes, isdiscs, labels)
decisionTree.visualize()