"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.11.10
- Brief: Test NeuralNetwork with watermelon dataset 3.0, 
and compare standard BP and accumulated BP. 
"""

import re
v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg

import time
import xlrd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from NeuralNetwork import NN

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
attributes = dataset[0][1:-1]
isdiscs = np.array([not bool(v.match(val)) for val in xs[0]])

# processing discrete values
for attr_idx in range(len(attributes)):
    if isdiscs[attr_idx]:
        values = list(set(xs[:, attr_idx]))
        xs[:, attr_idx] = np.array([values.index(val) for val in xs[:, attr_idx]])
xs = xs.astype(np.float64)

# partition dataset into train-set and test-set
train_indices = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
test_indices = [3, 4, 7, 8, 10, 11, 12]
train_xs, train_ys = xs[train_indices], ys[train_indices]
test_xs, test_ys = xs[test_indices], ys[test_indices]

epochs = 100

# Standard BP
print("### Standard BP ###")
nn = NN([xs.shape[1], 8, len(set(ys))], ["relu", "softmax"], lr_init=0.05, regularization="L2")
stdBP_loss = []
start = time.time()
for epoch in tqdm(range(epochs)):
    this_epoch_losses = []
    for sample_xs, sample_ys in zip(train_xs, train_ys):
        nn.train(sample_xs.reshape(1, -1), sample_ys.reshape(-1))
        this_epoch_losses.append(nn.loss)
    stdBP_loss.append(np.mean(this_epoch_losses))
end = time.time()
stdBP_time = end - start
stdBP_acc = np.mean(np.argmax(nn.forward(test_xs), axis=-1)==test_ys)

# Accumulated BP
print("\n### Accumulated BP ###")
nn.reset()
acmlBP_loss = []
start = time.time()
for epoch in tqdm(range(epochs)):
    nn.train(train_xs, train_ys)
    acmlBP_loss.append(nn.loss)
end = time.time()
acmlBP_time = end - start
acmlBP_acc = np.mean(np.argmax(nn.forward(test_xs), axis=-1)==test_ys)

# plot loss
plt.figure()
plt.plot(stdBP_loss, linewidth=2, label="Standard BP")
plt.plot(acmlBP_loss, linewidth=2, label="Accumulated BP")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# res
print("\n### After %d epochs training ###"%epochs)
print("- Standard BP:    using time = %.3fs, loss = %.4f, accuracy = %.2f"%(stdBP_time, stdBP_loss[-1], stdBP_acc*100) + "%")
print("- Accumulated BP: using time = %.3fs, loss = %.4f, accuracy = %.2f"%(acmlBP_time, acmlBP_loss[-1], acmlBP_acc*100) + "%")
# nn.visualize()
