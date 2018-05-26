# 交叉测试流程

## folds

1. 交叉验证根据说话人划分为5组，
2. folds里面包含5组文件。fold{i}\_train 是 第{i}组交叉验证的训练集，fold{i}\_eva 是 第{i}组交叉验证的验证集，fold{i}\_test 是 第{i}组交叉验证的测试集。 整个系统根据fold{i}\_train 训练，根据fold{i}\_eva 验证，根据fold{i}\_test 测试。
3. fold{i}\_train 是 第{i}组交叉验证的训练集，fold{i}\_eva 是 第{i}组交叉验证的验证集 是由原来的fold{i}\_train分割，且保持类别比例
4. 实验数据仅包含4类情感，`neutral happy angry sad`，在folds文件里简写为`neu hap ang sad`；
5. fold{i}\_trainj经过了over-sampling


## generateCV.py

产生folds中的文件，不用关心
