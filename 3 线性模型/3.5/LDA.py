import xlrd
import numpy as np
import matplotlib.pyplot as plt

# load data
data = xlrd.open_workbook('./WTMLDataSet.xlsx')
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

# mean
mu1 = np.mean(positive_xs, axis=0)
mu0 = np.mean(negative_xs, axis=0)

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
x = np.arange(0, 0.15, 0.01)
y = w[1]/w[0]*x
plt.plot(x, y, c='m', label='Projection line')
# plot data
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', marker='o', s=30, label='Positive')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', marker='o', s=30, label='Negative')
# projection
for i in range(positive_xs.shape[0]):
    prolen = positive_xs[i].dot(w)      # length of projection
    newp = prolen * w
    plt.plot([positive_xs[i, 0], newp[0]], [positive_xs[i, 1], newp[1]], color='#00CED1', linewidth=0.5, linestyle='--')
    positive_xs[i] = newp
for i in range(negative_xs.shape[0]):
    prolen = negative_xs[i].dot(w)      # length of projection
    newp = prolen * w
    plt.plot([negative_xs[i, 0], newp[0]], [negative_xs[i, 1], newp[1]], color='#DC143C', linewidth=0.5, linestyle='--')
    negative_xs[i] = newp
# plot data (after projection)
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], c='#00CED1', marker='*', s=30, label='Positive (projection)')
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], c='#DC143C', marker='*', s=30, label='Negative (projection)')

plt.legend()
plt.show()