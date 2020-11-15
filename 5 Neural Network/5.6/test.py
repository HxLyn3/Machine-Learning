"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.11.15
- Brief: Compared normal BP with constant learning rate and revised BP with 
exponential decay learning rate on Heart Failure Clinical Records Data Set
@http://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records. 
"""

import time
import xlrd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from NeuralNetwork import NN

# load data
data = xlrd.open_workbook('./heart_failure_clinical_records_dataset.xlsx')
table = data.sheet_by_name('heart_failure_clinical_records')

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    dataset.append(line)
dataset = np.array(dataset)

xs = dataset[1:, 0:-1].astype(np.float64)
ys = dataset[1:, -1].astype(np.float64).astype(np.int32)

# normalize
xs = (xs-np.mean(xs, axis=0))/np.std(xs, axis=0)

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

# select train dataset and test dataset with ratio 8:2
train_xs = np.vstack((positive_xs[:-2*pslide], negative_xs[:-2*nslide]))
train_ys = np.concatenate((positive_ys[:-2*pslide], negative_ys[:-2*nslide]))
test_xs = np.vstack((positive_xs[-2*pslide:], negative_xs[-2*nslide:]))
test_ys = np.concatenate((positive_ys[-2*pslide:], negative_ys[-2*nslide:]))

epochs = 100

# constant lr
print("### Constant Learning Rate ###")
nn = NN([xs.shape[1], 64, 64, len(set(ys))], ["relu", "relu", "softmax"], lr_init=0.01, regularization="L2", regularization_lambda=0.1)
lr_const_BP_loss = []
for epoch in tqdm(range(epochs)):
    nn.train(train_xs, train_ys)
    lr_const_BP_loss.append(nn.loss)
lr_const_BP_acc = np.mean(np.argmax(nn.forward(test_xs), axis=-1)==test_ys)

# exponential decay lr
print("\n### Exponential Decay Learning Rate ###")
nn = NN([xs.shape[1], 64, 64, len(set(ys))], ["relu", "relu", "softmax"], lr_init=0.01, lr_decay=0.99, lr_min=0.0001, regularization="L2", regularization_lambda=0.1)
lr_decay_BP_loss = []
for epoch in tqdm(range(epochs)):
    nn.train(train_xs, train_ys)
    lr_decay_BP_loss.append(nn.loss)
    # learning rate decay
    nn.lr_update()
lr_decay_BP_acc = np.mean(np.argmax(nn.forward(test_xs), axis=-1)==test_ys)

# plot loss
plt.figure()
plt.plot(lr_const_BP_loss, linewidth=2, label="Constant lr")
plt.plot(lr_decay_BP_loss, linewidth=2, label="Exponential Decay lr")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# res
print("\n### After %d epochs training ###"%epochs)
print("- Constant lr:  loss = %.4f, accuracy = %.2f"%(lr_const_BP_loss[-1], lr_const_BP_acc*100) + "%")
print("- Exp-Decay lr: loss = %.4f, accuracy = %.2f"%(lr_decay_BP_loss[-1], lr_decay_BP_acc*100) + "%")
