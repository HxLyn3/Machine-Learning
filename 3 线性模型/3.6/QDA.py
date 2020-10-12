"""
- Author: Haoxin Lin
- Email: linhx36@outlook.com
- Date: 2020.09.22
- Brief: Quadratic Discriminant Analysis (using sklearn)
"""

import xlrd
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

data = xlrd.open_workbook('../WTMLDataSet_3.0alpha.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)
xs = dataset[1:, 1:3].astype(np.float64)
ys = dataset[1:, 3].astype(np.float64).astype(np.int32)

# train
classifier = QDA()
classifier.fit(xs, ys)

# error rate
prediction = classifier.predict(xs)
error_rate = np.sum(prediction!=ys)/ys.shape[0]
print("Error rate: %.2f"%error_rate)

# visualize
# partition
nx, ny = 100, 100
xx, yy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='BuPu')
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

# plot data
positive_xs = xs[ys==1]
negative_xs = xs[ys==0]
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', s=60, label='Great (positive)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', s=60, label='Awful (negative)')

plt.legend()
plt.show()