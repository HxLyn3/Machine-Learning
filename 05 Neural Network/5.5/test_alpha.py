"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.11.10
- Brief: Test NeuralNetwork with watermelon dataset 3.0α
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NN

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

# train a neural network to learn from watermelon dataset
nn = NN([xs.shape[1], 16, len(set(ys))], ["sigmoid", "softmax"], lr_init=0.1, regularization=None)
for batch_idx in range(50000):
    nn.train(xs, ys)
    if batch_idx % 100 == 0:
        print("Loss = %.4f"%nn.loss)

# calculate accuracy
preds = nn.forward(xs)
preds = np.argmax(preds, axis=-1)
print("Accuracy: %.4f"%np.mean(preds==ys))

# plot data
positive_xs = xs[ys==1]
negative_xs = xs[ys==0]
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', s=60, label='Great (positive)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', s=60, label='Awful (negative)')

# partition
x0, x1 = 1000, 1000
x0, x1 = np.meshgrid(np.linspace(0, 1, x0), np.linspace(0, 1, x1))
xs = np.concatenate([x0.reshape(1000000, 1), x1.reshape(1000000, 1)], axis=-1)
preds = np.argmax(nn.forward(xs), axis=-1).reshape(x0.shape)
plt.contour(x0, x1, preds, [0.5], linewidths=2, colors='m')

plt.xlabel("x[0]: Density")
plt.ylabel("x[1]: Sugar Content")
plt.legend()
plt.show()