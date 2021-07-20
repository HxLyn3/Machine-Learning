# Problem 5.7
根据式(5.18)和(5.19)，试构造一个能解决异或问题的单层RBF神经网络。

## Environment
- `python 3.7.0`  
- `xlrd 1.2.0`  
- `numpy 1.19.2`  
- `matplotlib 3.3.2`  

## Usage
```Shell
python3 test.py
```

## Result
Outcome of RBF Network at XOR problem:  
![image](./output.png)  

## P.S.
In order to visualize RBF Network, I also train RBF Network with watermelon dataset 3.0α whose data only have two attributes. The cmd is ```python test_alpha.py```. Then you can see the boundary that RBF Network draws on dataset.  
![image](./output_3.0alpha.png)  