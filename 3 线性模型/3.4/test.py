"""
- Author: Haoxin Lin
- Email: linhx36@outlook.com
- Date: 2020.09.21
- Brief: Compare 10-fold cross validation and Leave-One-Out
"""

import xlrd
import numpy as np
from LogitReg import LogitReg

# load dataset
data = xlrd.open_workbook('./SomervilleHappinessSurvey2015.xlsx')
table = data.sheet_by_index(0)

dataset = []
for i in range(table.nrows):
    line = table.row_values(i)
    line = line[0].split(',')
    dataset.append(line)
dataset = np.array(dataset)

# extract dataset
xs = dataset[1:, 1:].astype(np.float64)
ys = dataset[1:, 0].astype(np.int32)

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

# classifier
classifier = LogitReg(xsize=positive_xs.shape[1])

""" 10-fold cross validation """
print("10-fold Cross Validation Test")
pselect = np.ones(positive_xs.shape[0])
nselect = np.ones(negative_xs.shape[0])

total_error_rate1 = 0

for i in range(10):
    pselect[pslide*i:pslide*(i+1)] = 0
    nselect[nslide*i:nslide*(i+1)] = 0

    # dataset segmentation
    train_xs = np.vstack((positive_xs[pselect==1], negative_xs[nselect==1]))
    train_ys = np.hstack((positive_ys[pselect==1], negative_ys[nselect==1]))
    test_xs = np.vstack((positive_xs[pselect==0], negative_xs[nselect==0]))
    test_ys = np.hstack((positive_ys[pselect==0], negative_ys[nselect==0]))

    # train
    print("Dataset Segmentation %d"%i)
    classifier.load(train_xs, train_ys)
    classifier.learn()

    # test
    _, error_rate = classifier.test(test_xs, test_ys)
    print("Error rate: %.2f"%error_rate)
    total_error_rate1 += error_rate

    # reset
    classifier.reset()
    pselect[pslide*i:pslide*(i+1)] = 1
    nselect[nslide*i:nslide*(i+1)] = 1
    print()

# result of test
total_error_rate1 /= 10
print("Total error rate of 10-fold cross validation is: %.2f"%total_error_rate1)
print()



""" Leave One Out """
print("Leave-One-Out Test")
xs = np.vstack((positive_xs, negative_xs))
ys = np.hstack((positive_ys, negative_ys))
select = np.ones(xs.shape[0])

total_error_rate2 = 0
for i in range(xs.shape[0]):
    select[i] = 0

    # dataset segmentation
    train_xs = xs[select==1]
    train_ys = ys[select==1]
    test_xs = xs[select==0]
    test_ys = ys[select==0]

    # train
    print("Dataset Segmentation %d"%i)
    classifier.load(train_xs, train_ys)
    classifier.learn()

    # test
    _, error_rate = classifier.test(test_xs, test_ys)
    print("Error rate: %.2f"%error_rate)
    total_error_rate2 += error_rate

    # reset
    classifier.reset()
    select[i] = 1
    print()

# result of test
total_error_rate2 /= xs.shape[0]
print("Total error rate of Leave-One-Out is: %.2f"%total_error_rate2)


# comparison
print("Comparison:")
print("10-fold Cross Validation: %.2f"%total_error_rate1)
print("Leave-One-Out: %.2f"%total_error_rate2)