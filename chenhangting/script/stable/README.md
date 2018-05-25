# 基线DNN和LSTM

## 流程

1. 将原来的训练集`AudioEMOTION\CV\fold{i}_train.txt`,根据各标签的比例划分成`fold{i}_train.txt`和`fold{i}_eva.txt`,前者用来训练，后者用来early stopping
2. 根据`AudioEMOTION\chenhangting\CV\foldsfold{i}_train.txt`训练
3. 根据`AudioEMOTION\chenhangting\CV\foldsfold{i}_eva.txt`调节超参
4. 根据`AudioEMOTION\chenhangting\CV\foldsfold{i}_test.txt`得到测试结果

## 特征

lzm的特征，依照文献[1]的描述

## 神经网络

1. 4*ReLU256,`baselinednn.py`
2. 2*BiLSTM256,`baselinelstm.py`
3. 2*BiLSTM256+attention1,`attention1.py`
4. 2*BiLSTM256+attention2,`attention2.py`
5. 2*BiLSTM256+attention3,`attention3.py`

## 实验结果

| No. | Net | Feature | Training Method | Macro F-score(%) | Acc(%) |
| :-  | -   |       - | -               |  -:            |  -: |
| 1 | dnn | lzm | frame-wise weighted loss | 50.2 | 55.6 |
| 2 | LSTM | lzm | frame-wise weighted loss | 53.7 | 57.6 |
| 3 | attention1 | lzm | segment-based weighted loss | 53.3 | 57.0 |
| 4 | attention2 | lzm | segment-based weighted loss | 54.3 | 57.6 |
| 5 | attention3 | lzm | segment-based weighted loss | 55.3 | 58.9 |
| 6 | Advanced-LSTM[1] | Seq(MFCC+others) | segment-based unknown | 46.2 | 55.3 |
| 7 | dnn[1] | IS10 | segment_based unknown | 56.9 | 58.2 |
| 8 | Advanced-LSTM+dnn[1] | IS10+Seq | fusion | 58.2 | 58.7 |
| 9 | rnn-attention1[2] | MFCC+others | segment-based weighted loss | - | 63.5 |


1. attention1(也就是[1]和[2]中使用的)，效果成谜。至少目前看来没有积极作用
2. attention3看起来是有效果的，但是我也不确定这个提升是否存在偶然性；attention3在fscore上与 No.7 dnn+IS10相距甚大；attention3的acc表现不错
3. No.9[2]，用了简单的attention1，但是acc有63.5%，所以它是怎么做到的


## Ref.

[1] Tao, F. and G. Liu (2017). "Advanced LSTM: A Study about Better Time Dependency Modeling in Emotion Recognition."

[2] Mirsamadi, S., et al. (2017). Automatic speech emotion recognition using recurrent neural networks with local attention. IEEE International Conference on Acoustics, Speech and Signal Processing.