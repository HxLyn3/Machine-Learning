# Problem 3.6
线性判别分析仅在线性可分数据上能获得理想结果，试设计一个改进方法，使其能较好地用于非线性可分数据。  
  
Use sklearn to implement QDA, just for experiment.

## Dataset
西瓜数据集3.0α @ `../WTMLDataSet_3.0alpha.xlsx`
编号|	密度|	含糖率|	好瓜
|--| --|--|--|
1|	0.697|	0.46|	1|
2|	0.774|	0.376|	1|
3|	0.634|	0.264|	1|
4|	0.608|	0.318|	1|
5|	0.556|	0.215|	1|
6|	0.403|	0.237|	1|
7|	0.481|	0.149|	1|
8|	0.437|	0.211|	1|
9|	0.666|	0.091|	0|
10|	0.243|	0.267|	0|
11|	0.245|	0.057|	0|
12|	0.343|	0.099|	0|
13|	0.639|	0.161|	0|
14|	0.657|	0.198|	0|
15|	0.36|	0.37|	0|
16|	0.593|	0.042|	0|
17|	0.719|	0.103|	0|

## Environment
- `python 3.5.4`  
- `xlrd 1.2.0`  
- `numpy 1.15.4`
- `matplotlib 3.0.0`
- `sklearn 0.22.2.post1`

## Usage
```Shell
python3 QDA.py
```

## Result
![image](./output.png)  
QDA is able to deal with the nonlinear sparable dataset well.