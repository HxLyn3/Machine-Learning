"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/12/14
- Brief: Test Support Vector Regression
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from SVR import SVR

# create and plot data
xs = np.arange(0, 6*math.pi, 0.2).reshape(-1, 1)
ys = np.sin(xs)

# fit with Support Vector Regression
svr = SVR(xs.shape[1], func='Gaussian', sigma=0.1)
svr.fit(xs, ys, C=10, epsilon=0.1, err=0.001, iters=10)

# plot fitted curve
preds = svr.predict(xs)
plt.plot(xs, preds, linestyle='--', c='skyblue', label='fit curve')
# plot source data
plt.scatter(xs, ys, c='#AC5EB8', marker='x', s=20, label='sin(data)')

plt.axis([0, 21, -1.1, 1.1])
plt.legend()
plt.show()
