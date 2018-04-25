# 基线DNN和LSTM

## 流程

1. 将原来的训练集`AudioEMOTION\CV\fold{i}_train.txt`,根据各标签的比例划分成`fold{i}_train.txt`和`fold{i}_eva.txt`,前者用来训练，后者用来early stopping
2. 根据`AudioEMOTION\chenhangting\CV\foldsfold{i}_train.txt`训练
3. 根据`AudioEMOTION\chenhangting\CV\foldsfold{i}_eva.txt`调节超参
4. 根据`AudioEMOTION\chenhangting\CV\foldsfold{i}_test.txt`得到测试结果

## 特征
1. 25ms帧长，10ms帧移
2. 40个mel滤波器
3. 加入能量、谱重心、过零率、子带能量（8个子带），共51维
4. 加入delta和delta-delta，共51*3=153维
5. 脚本路径 `AudioEMOTION\chenhangting\script\features\fbankExtract.sh`

## 神经网络
1. 4*256ReLu,`AudioEMOTION\chenhangting\script\lstmBaseline\baselinednn_early_stopping.py`
2. 4*256ReLu+weight,`AudioEMOTION\chenhangting\script\lstmBaseline\baselinednn_weight_early_stopping.py`
3. 4*256lstm,`AudioEMOTION\chenhangting\script\lstmBaseline\baselinelstm_early_stopping.py`
4. 4*256lstm+weight,`AudioEMOTION\chenhangting\script\lstmBaseline\baselinelstm_weight_early_stopping.py`


## 实验结果

| Net | Macro F-score | Acc |
| :- | - | -: |
| dnn | 0.484 | 0.596 |
| dnn+weight | 0.530 | 0.572 |
| lstm | 0.506 | 0.558 |
| lstm+weight | 0.515 | 0.552 |

