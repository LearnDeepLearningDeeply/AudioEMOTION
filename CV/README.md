# 交叉验证流程

## folds

1. 交叉验证根据说话人划分为5组，
2. folds里面包含5组文件。fold{i}\_train 是 第{i}组交叉验证的训练集，fold{i}\_test 是 第{i}组交叉验证的测试集。 整个系统根据fold{i}\_train 训练，根据fold{i}\_test 测试。
3. 实验数据仅包含4类情感，`neutral happy angry sad`，在folds文件里简写为`neu hap ang sad`；共有样本4490条

## generateCV.py

产生folds中的文件，不用关心