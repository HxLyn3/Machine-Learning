"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/11/23
- Brief: Solve nonlinear xor problem with RBF Network
"""

import numpy as np
import matplotlib.pyplot as plt
from MonoRBF import MonoRBF

x0, x1 = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
x0 = x0.reshape(-1, 1)
x1 = x1.reshape(-1 ,1)
xs = np.concatenate([x0, x1], axis=-1)
ys = (x0.astype(np.int32)^x1.astype(np.int32)).astype(np.int32).reshape(-1)

# RBF Network
RBFNet = MonoRBF(hidden_size=4, n_classes=2)
RBFNet.load_training(xs, ys)
RBFNet.learn(lr=0.5, epochs=1000)

# calculate accuracy
preds = RBFNet.forward(xs)
preds = np.argmax(preds, axis=-1)
print("Accuracy: %.4f"%np.mean(preds==ys))

# plot data
positive_xs = xs[ys==1]
negative_xs = xs[ys==0]
plt.scatter(positive_xs[:, 0], positive_xs[:, 1], marker='+', c='#00CED1', s=100)
plt.scatter(negative_xs[:, 0], negative_xs[:, 1], marker='_', c='#DC143C', s=100)

# partition
x0, x1 = 1000, 1000
x0, x1 = np.meshgrid(np.linspace(-0.5, 1.5, x0), np.linspace(-0.5, 1.5, x1))
xs = np.concatenate([x0.reshape(1000000, 1), x1.reshape(1000000, 1)], axis=-1)
preds = np.argmax(RBFNet.forward(xs), axis=-1).reshape(x0.shape)
plt.contour(x0, x1, preds, [0.5], linewidths=2, colors='m')

plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.title("XOR")
plt.show()