"""
- Author: Haoxin Lin
- Email: linhx36@outlook.com
- Date: 2020.09.22
- Brief: Linear Discriminant Analysis
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt

""" use watermelon dataset """
"""
# load data
data = xlrd.open_workbook('../WTMLDataSet_3.0alpha.xlsx')
table = data.sheet_by_name('WTML')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)
xs = dataset[1:, 1:3].astype(np.float64)
ys = dataset[1:, 3].astype(np.float64).astype(np.int32)

positive_xs = xs[ys==1]
positive_ys = ys[ys==1]
negative_xs = xs[ys==0]
negative_ys = ys[ys==0]
"""

# create dataset
positive_xs = np.random.randn(16, 2)*0.1 + 0.7
negative_xs = np.random.randn(16, 2)*0.1 + 0.3

# mean
mu1 = np.mean(positive_xs, axis=0)
mu0 = np.mean(negative_xs, axis=0)
mu = (mu0+mu1)/2

# covariance
Sigma1 = np.cov(positive_xs.T)
Sigma0 = np.cov(negative_xs.T)

Sw = Sigma0 + Sigma1
Sb = np.matmul((mu0-mu1)[:, np.newaxis], (mu0-mu1)[np.newaxis, :])

# max J = |wT*mu0-wT*mu1|^2 / wT*(Sigma0+Sigma1)*w
# equal to    min    -wT*Sb*w
#             s.t.   wT*Sw*w = 1
# Lagrange    f(w,λ) = -wT*Sb*w + λ*(wT*Sw*w-1)
#             df/dw  = 2*(λ*Sw*w-Sb*w) = 0
#                    ==> Sb*w = λ*Sw*w
#             Sb*w = (mu0-mu1)*(mu0-mu1)T*w = C*(mu0-mu1)
#                w = C*Inv(Sw)*(mu0-mu1)

# res 
w = np.matmul(np.linalg.inv(Sw), (mu0-mu1)[:, np.newaxis])
w = w[:, 0]
w /= np.sqrt((w.dot(w)))

# visualize

# projection line
k = w[1]/w[0]
x = np.arange(0, 1, 0.0001)
y = k*(x-mu[0])+mu[1]
plt.plot(x, y, c='m', label='Projection line')

# plot data
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', marker='o', s=30, label='Positive')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', marker='o', s=30, label='Negative')

# projection
for i in range(positive_xs.shape[0]):
    dis = (positive_xs[i,1]-k*positive_xs[i,0]+k*mu[0]-mu[1])/np.sqrt(k**2+1) # distance from point to line
    newp = positive_xs[i].copy()
    newp[0] -= dis*w[1]
    newp[1] += dis*w[0]
    plt.plot([positive_xs[i, 0], newp[0]], [positive_xs[i, 1], newp[1]], color='#00CED1', linewidth=0.5, linestyle='--')
    positive_xs[i] = newp

for i in range(negative_xs.shape[0]):
    dis = (negative_xs[i,1]-k*negative_xs[i,0]+k*mu[0]-mu[1])/np.sqrt(k**2+1) # distance from point to line
    newp = negative_xs[i].copy()
    newp[0] -= dis*w[1]
    newp[1] += dis*w[0]
    plt.plot([negative_xs[i, 0], newp[0]], [negative_xs[i, 1], newp[1]], color='#DC143C', linewidth=0.5, linestyle='--')
    negative_xs[i] = newp

# plot data (after projection)
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', marker='*', s=30, label='Positive (projection)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', marker='*', s=30, label='Negative (projection)')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.show()