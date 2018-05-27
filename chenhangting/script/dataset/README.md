# Dataset Loader in Pytorch

## Notes

1. 使用前请先安装progressbar2和pytorch。 pip install progressbar2
2. torch.utils.data.Dataset 是关于dataset的一个类. 自己自定义的 Dataset 应该包括:`__len__` 和`__getitem__` 方法

## dataset1d_early_stopping.py

1. 支持`train`,`eva`,`test`三种模式
2. 只能用于帧级特征，在加载过程中会自动补零至最大长度
3. 输出标签为帧级标签

## dataset1d_early_stopping_single_label.py

1. 支持`train`,`eva`,`test`三种模式
2. 只能用于帧级特征，在加载过程中会自动补零至最大长度
3. 输出标签为音频级标签

## dataset1d_longterm.py

1. 支持`train`,`eva`,`test`三种模式
2. 只能用于长时特征
3. 输出标签为音频级标签

## 使用方法（以dataset1d_longterm.py为例）

```python
#dataset1d_longterm.py 应该在你的路径下
from dataset1d_longterm import AudioFeatureDataset

# 加载训练数据
trainDataset=AudioFeatureDataset(featrootdir=r'../../data/data',cvtxtrootdir=r'../../CV/folds',feattype='csv',cvnum=1,mode='train',normflag=1,normfile='ms1.npy')
#featrootdir是特征所在的文件夹
#feattype是特征格式，目前支持'csv','npy'以及'txt';可以在函数npload中自由添加
#cvnum 折数
#mode 模式='train','eva','test'
#normflag =0 依据当前数据集计算mean和std，保存在normfile位置，并且在数据集上做归一化处理；=1 读取保存在normfile位置的归一化文件，并在数据集上做归一化处理；=-1 不做任何归一化，此时normfile参数无效

# 加载验证数据
evaDataset=AudioFeatureDataset(featrootdir=r'../../data/data',cvtxtrootdir=r'../../CV/folds',feattype='csv',cvnum=1,mode='eva',normflag=0,normfile='ms1.npy')

# 加载测试数据
testDataset=AudioFeatureDataset(featrootdir=r'../../data/data',cvtxtrootdir=r'../../CV/folds',feattype='csv',cvnum=1,mode='test',normflag=0,normfile='ms1.npy')
````