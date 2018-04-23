# 交叉测试流程

## folds

1. 交叉验证根据说话人划分为5组，
2. folds里面包含5组文件。fold{i}\_train 是 第{i}组交叉验证的训练集，fold{i}\_test 是 第{i}组交叉验证的测试集。 整个系统根据fold{i}\_train 训练，根据fold{i}\_test 测试。
3. 实验数据仅包含4类情感，`neutral happy angry sad`，在folds文件里简写为`neu hap ang sad`；共有样本4490条
4. 本交叉验证集的设定参考 Tao, F. and G. Liu (2017). "Advanced LSTM: A Study about Better Time Dependency Modeling in Emotion Recognition."

| samples num in fold{i}(train/test) | neu | hap | ang | sad |
| - | :-: | :-: | :-: | -: | 
| 1 | 1324/384| 460/135 | 874/229 | 890/194 | 
| 2 | 1346/362 | 478/117 | 966/137 | 887/197 |
| 3 | 1388/320 | 460/135 | 863/240 | 779/305 |
| 4 | 1450/258 | 530/65 | 776/327 |941/143 |
| 5 | 1324/384 | 452/143 | 933/170 | 839/245 |

## generateCV.py

产生folds中的文件，不用关心
