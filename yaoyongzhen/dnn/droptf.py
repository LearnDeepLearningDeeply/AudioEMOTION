"""
@date: Created on 2018/5/30
@author: yaoyongzhen
@notes: 去掉frametime，对label进行编码，去掉文件头
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dropft(input_path,out_path):
    data = pd.read_csv(input_path)
    data.pop('frameTime')
    data['label'] = lb.transform(list(data['label']))
    data.to_csv(out_path,index=False,header=False)

print(1)
data = pd.read_csv('fold1_test.txt')
lb = LabelEncoder()#对不连续的数字或者文本进行编号
lb.fit((list(data['label'])))
dropft('fold1_test.txt','dropft/fold1_test.txt')
dropft('fold1_train.txt','dropft/fold1_train.txt')
print(2)
dropft('fold2_test.txt','dropft/fold2_test.txt')
dropft('fold2_train.txt','dropft/fold2_train.txt')
print(3)
dropft('fold3_test.txt','dropft/fold3_test.txt')
dropft('fold3_train.txt','dropft/fold3_train.txt')
print(4)
dropft('fold4_test.txt','dropft/fold4_test.txt')
dropft('fold4_train.txt','dropft/fold4_train.txt')
print(5)
dropft('fold5_test.txt','dropft/fold5_test.txt')
dropft('fold5_train.txt','dropft/fold5_train.txt')
