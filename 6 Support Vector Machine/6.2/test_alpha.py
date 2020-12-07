"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.12.07
- Brief: Test Linear-SVM and Gaussian-SVM with watermelon dataset 3.0α
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM

plt.figure()

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
positive_xs, negative_xs = xs[ys==1], xs[ys==0]
ys = np.where(ys==1, 1, -1)

# Linear SVM
Linear_SVM = SVM(xs.shape[1], func='Linear')
Linear_SVM.fit(xs, ys, C=100, epsilon=0.01, iters=1000)

# Gaussian SVM
Gaussian_SVM = SVM(xs.shape[1], func='Gaussian', sigma=0.1)
Gaussian_SVM.fit(xs, ys, iters=1000)

# meshgrid data
x0, x1 = 1000, 1000
x0, x1 = np.meshgrid(np.linspace(0, 1, x0), np.linspace(0, 1, x1))
xs = np.concatenate([x0.reshape(1000000, 1), x1.reshape(1000000, 1)], axis=-1)

# plot outcome of Linear SVM
plt.subplot(1, 2, 1)
plt.scatter(Linear_SVM.support_vectors[:, 0], Linear_SVM.support_vectors[:, 1], marker='o', c='w', edgecolors='orange', s=150, label='Support Vector')
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', s=60, label='Great (positive)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', s=60, label='Awful (negative)')

preds = Linear_SVM.predict(xs).reshape(x0.shape)
plt.contour(x0, x1, preds, [0.5], linewidths=2, colors='m')
plt.xlabel("x[0]: Density"); plt.ylabel("x[1]: Sugar Content"); plt.title("Outcome of Linear SVM")
plt.legend()

# plot outcome of Gaussian SVM
plt.subplot(1, 2, 2)
plt.scatter(Gaussian_SVM.support_vectors[:, 0], Gaussian_SVM.support_vectors[:, 1], marker='o', c='w', edgecolors='orange', s=150, label='Support Vector')
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', s=60, label='Great (positive)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', s=60, label='Awful (negative)')

preds = Gaussian_SVM.predict(xs).reshape(x0.shape)
plt.contour(x0, x1, preds, [0.5], linewidths=2, colors='m')
plt.xlabel("x[0]: Density"); plt.ylabel("x[1]: Sugar Content"); plt.title("Outcome of Gaussian SVM")
plt.legend()

plt.show()
