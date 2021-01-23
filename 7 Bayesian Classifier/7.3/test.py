"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2021/01/23
- Brief: build naive Bayes classifier with watermelon dataset 3.0
"""

import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import xlrd
import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier

# load data
data = xlrd.open_workbook('../WTMLDataSet_3.0.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)

xs = dataset[1:, 1:-1]
ys = (dataset[1:, -1]=='否').astype(np.int32)
isdiscs = np.array([not bool(v.match(val)) for val in xs[0]])
labels = ['好瓜', '坏瓜']

# build naive Bayes classifier
classifier = NaiveBayesClassifier(xs, isdiscs, ys)

# input
test_x = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]
print("Input: \t%ls"%test_x)

# output
test_y = classifier.classify(np.array(test_x))
print("Output: %s"%labels[test_y])