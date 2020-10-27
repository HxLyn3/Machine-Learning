"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.10.21
- Brief:
    Decision Tree Test Table (on Cryotherapy Dataset):
    |                  | No-pruning | Pre-pruning | Post-pruning
    | Information Gain | 0.7500     | 1.0000      | 1.0000
    | Gini Index       | 0.8750     | 1.0000      | 1.0000
    | Logit Regression | 0.8750     | 1.0000      | 1.0000
"""
import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
from DecisionTree_v3 import DecisionTree

# load data
data = xlrd.open_workbook('../CrayoDataset.xlsx')
table = data.sheet_by_name('CrayoDataset')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    if not '' in line:
        dataset.append(line)
dataset = np.array(dataset)

xs = dataset[1:, 0:-1]
ys = (dataset[1:, -1]=='1.0').astype(np.int32)

# dataset division
positive_xs = xs[ys==1]
positive_ys = ys[ys==1]
negative_xs = xs[ys==0]
negative_ys = ys[ys==0]

pslide = positive_xs.shape[0] // 10
pres = positive_xs.shape[0] % 10
nslide = negative_xs.shape[0] // 10
nres = negative_xs.shape[0] % 10

positive_xs = positive_xs[:-pres]
positive_ys = positive_ys[:-pres]
negative_xs = negative_xs[:-nres]
negative_ys = negative_ys[:-nres]

# select train dataset and test dataset with ratio 9:1
train_xs = np.vstack((positive_xs[:-pslide], negative_xs[:-nslide]))
train_ys = np.concatenate((positive_ys[:-pslide], negative_ys[:-nslide]))
test_xs = np.vstack((positive_xs[-pslide:], negative_xs[-nslide:]))
test_ys = np.concatenate((positive_ys[-pslide:], negative_ys[-nslide:]))

attributes = dataset[0][0:-1]
isdiscs = np.array([True, False, False, False, True, False])
labels = ["Failed", "Successful"]

indices = ['InformationGain', 'GiniIndex', 'LogitRegression']
for index in indices:
    # no pruning
    print("Decision Tree with index of %s (No Pruning). Constructing..."%index)
    decisionTree = DecisionTree(train_xs, train_ys, test_xs, test_ys, attributes, isdiscs, labels)
    decisionTree.buildTree(partIndex=index, prepruning=False)
    decisionTree.visualize(graph_name=index + "_No-Pruning")
    print("Accuracy: %.4f\n"%decisionTree.test(test_xs, test_ys))

    # post-pruning
    decisionTree.post_pruning()
    decisionTree.visualize(graph_name=index + "_Post-Pruning")
    print("Accuracy: %.4f\n"%decisionTree.test(test_xs, test_ys))

    # pre-pruning
    print("Decision Tree with index of %s (Pre-Pruning). Constructing..."%index)
    decisionTree.buildTree(partIndex=index, prepruning=True)
    decisionTree.visualize(graph_name=index + "_Pre-Pruning")
    print("Accuracy: %.4f\n"%decisionTree.test(test_xs, test_ys))