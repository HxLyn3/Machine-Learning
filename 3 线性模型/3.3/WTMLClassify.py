import xlrd
import numpy as np
from LogitReg import LogitReg

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

# Logit Regression
classifier = LogitReg(xsize=xs.shape[1])
classifier.load(xs, ys)
classifier.learn()
_, error_rate = classifier.test(xs, ys)
print("Error Rate: %.2f"%error_rate)
classifier.visualize()