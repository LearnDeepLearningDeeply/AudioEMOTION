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

| Net | Feature | Macro F-score | Acc |
| :- | - | - | -: |
| dnn | fbank+others | 0.484 | 0.596 |
| dnn+weight | fbank+others | 0.530 | 0.572 |
| lstm | fbank+others | 0.506 | 0.558 |
| lstm+weight | fbank+others | 0.515 | 0.552 |
| Advanced-LSTM[1] | Seq(MFCC+others) | 0.462 | 0.553 |
| dnn[1] | IS10 | 0.569 | 0.582 |
| Advanced-LSTM+dnn[1] | IS10+Seq | 0.582 | 0.587 |
| rnn-attention[2] | MFCC+others | - | 0.588 |


1. 总体来看，针对frame-wise的预测，dnn表现更好
2. 针对不同类别加入weight以平衡的方法可以提升`Macro F-score`，令`Acc`下降
3. [1]和[2]似乎都没有加入weight来平衡类别
4. [1]采用了utterance-level的归一化，[2]采用了全局归一化，基线系统使用全局归一化


## Ref.

[1] Tao, F. and G. Liu (2017). "Advanced LSTM: A Study about Better Time Dependency Modeling in Emotion Recognition."

[2] Mirsamadi, S., et al. (2017). Automatic speech emotion recognition using recurrent neural networks with local attention. IEEE International Conference on Acoustics, Speech and Signal Processing.