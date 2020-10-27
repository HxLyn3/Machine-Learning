"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.10.20
- Brief: Test decision tree which select Logit Regression as index to choose optimal
attribute for partition on watermelon dataset 3.0.
"""
import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
from DecisionTree_v3 import DecisionTree

# load data
data = xlrd.open_workbook('../WTMLDataSet_3.0.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)
dataset = dataset[:, [0,5,1,2,3,4,6,7,8,9]]     # adjust attributes priority

xs = dataset[1:, 1:-1]
ys = (dataset[1:, -1]=='否').astype(np.int32)
attributes = dataset[0][1:-1]
isdiscs = np.array([not bool(v.match(val)) for val in xs[0]])
labels = ["好瓜", "坏瓜"]

# Logit Regression Decision Tree
decisionTree = DecisionTree(xs, ys, xs, ys, attributes, isdiscs, labels)
decisionTree.buildTree(partIndex='LogitRegression', prepruning=False)
decisionTree.visualize(graph_name="DecisionTree")
