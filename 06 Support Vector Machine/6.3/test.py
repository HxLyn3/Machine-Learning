"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.12.11
- Brief: Compared Linear SVM, Gaussian SVM, Neural Network and Decision Tree on Heart Failure 
Clinical Records Data Set @http://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records. 
"""

import time
import xlrd
import numpy as np
from tqdm import tqdm
from SVM import SVM
from NeuralNetwork import NN
from DecisionTree_v2 import DecisionTree

# load data
data = xlrd.open_workbook('./heart_failure_clinical_records_dataset.xlsx')
table = data.sheet_by_name('heart_failure_clinical_records')

dataset = []
for i in range(table.nrows):
    dataset.append(table.row_values(i))
dataset = np.array(dataset)

xs = dataset[1:, 0:-1].astype(np.float64)
ys = dataset[1:, -1].astype(np.float64).astype(np.int32)
attributes = dataset[0][1:-1]; isdiscs = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]); labels = ["no", "yes"]
# normalize
xs = (xs-np.mean(xs, axis=0))/np.std(xs, axis=0)

# shuffle
indices = np.random.choice(xs.shape[0], xs.shape[0], replace=False)
xs = xs[indices]; ys = ys[indices]

# dataset division
positive_xs = xs[ys==1]
positive_ys = ys[ys==1]
negative_xs = xs[ys==0]
negative_ys = ys[ys==0]
pslide = positive_xs.shape[0] // 10
nslide = negative_xs.shape[0] // 10

# select train dataset and test dataset with ratio 8:2
train_xs = np.vstack((positive_xs[:-2*pslide], negative_xs[:-2*nslide]))
train_ys = np.concatenate((positive_ys[:-2*pslide], negative_ys[:-2*nslide]))
test_xs = np.vstack((positive_xs[-2*pslide:], negative_xs[-2*nslide:]))
test_ys = np.concatenate((positive_ys[-2*pslide:], negative_ys[-2*nslide:]))

train_ys_for_svm = np.where(train_ys==0, -1, 1)
test_ys_for_svm = np.where(test_ys==0, -1, 1)
# Linear SVM
print("\nTesting SVM with linear kernel...")
Linear_SVM = SVM(xs.shape[1], func='Linear')
Linear_SVM.fit(train_xs, train_ys_for_svm, C=100, epsilon=0.01, iters=10000)
Linear_svm_acc = np.mean(Linear_SVM.predict(test_xs)==test_ys_for_svm)

# Gaussian SVM
print("\nTesting SVM with Gaussian kernel...")
Gaussian_SVM = SVM(xs.shape[1], func='Gaussian', sigma=0.1)
Gaussian_SVM.fit(train_xs, train_ys_for_svm, C=1, epsilon=0.01, iters=100)
Gaussian_svm_acc = np.mean(Gaussian_SVM.predict(test_xs)==test_ys_for_svm)

# Neural Network
print("\nTesting Neural Network...")
nn = NN([xs.shape[1], 64, len(set(ys))], ["relu", "softmax"], lr_init=0.01, regularization="L2", regularization_lambda=0.1)
for epoch in tqdm(range(100)): nn.train(train_xs, train_ys)
nn_acc = np.mean(np.argmax(nn.forward(test_xs), axis=-1)==test_ys)

# Decision Tree
print("\nTesting Decision Tree...")
decisionTree = DecisionTree(train_xs, train_ys, test_xs, test_ys, attributes, isdiscs, labels)
decisionTree.buildTree(partIndex='InformationGain', prepruning=True)
decisionTree_acc = decisionTree.test(test_xs, test_ys)

# Demo
print("\nTest Accuracy:")
print("- Linear SVM      :    %.2f"%(Linear_svm_acc*100)+"%")
print("- Gaussian SVM    :    %.2f"%(Gaussian_svm_acc*100)+"%")
print("- Neural Network  :    %.2f"%(nn_acc*100)+"%")
print("- Decision Tree   :    %.2f"%(decisionTree_acc*100)+"%")
