# Problem 6.3
选择两个UCI数据集，分别用线性核和高斯核训练一个SVM，并与BP神经网络和C4.5决策树进行实验比较。（UCI数据集见<http://archive.ics.uci.edu/ml/>.）

## Dataset
UCI 心衰临床记录数据集 @ `./heart_failure_clinical_records_dataset.xlsx`

## Environment
- `python 3.7.0`  
- `xlrd 1.2.0`  
- `tqdm 4.50.2`  
- `numpy 1.19.2`  

## Usage
```Shell
python3 test.py
```

## Comparison  
```Python
Test Accuracy:
- Linear SVM      :    68.97%
- Gaussian SVM    :    81.03%
- Neural Network  :    79.31%
- Decision Tree   :    79.31%
```  
- Consumption: SVM and Neural Network require more computing resources than Decision Tree.
- Parameters Tuning: important for Neural Network, but not for SVM and Decision Tree.
- Performance: The above algorithms could deal with nonlinear separable dataset except linear SVM. Gaussian SVM, Neural Network and Decision Tree perform similarly on low-dimensional and small dataset (such as heart_failure_clinical_records_dataset, only 12 dimensions and hundreds of pieces of data).