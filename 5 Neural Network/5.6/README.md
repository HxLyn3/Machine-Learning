# Problem 5.6
试设计一个BP改进算法，能通过动态调整学习率显著提升收敛速度。编程实现该算法，并选择两个UCI数据集与标准BP算法进行实验比较。（UCI数据集见<http://archive.ics.uci.edu/ml/>.）

## Dataset
UCI 心衰临床记录数据集 @ `./heart_failure_clinical_records_dataset.xlsx`

## Environment
- `python 3.7.0`  
- `xlrd 1.2.0`  
- `tqdm 4.50.2`  
- `numpy 1.19.2`  
- `matplotlib 3.3.2`  
- `graphviz 0.14.2`  

## Usage
```Shell
python3 test.py
```

## Comparison  
Training 100 epochs.  
![image](./output.png)  
```Python
### After 100 epochs training ###
- Constant lr:  loss = 0.5539, accuracy = 72.41%
- Exp-Decay lr: loss = 0.4894, accuracy = 74.14%
```  
BP algorithm with exponential decay learning rate makes loss converges faster.