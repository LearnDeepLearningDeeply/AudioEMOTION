"""
@date: Created on 2018/5/30
@author: yaoyongzhen
@notes: DNN模型
"""
#加载包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn import metrics

# 数据集名称
TRAINING = "folds/dropft/fold1_train.txt"
TEST = "folds/dropft/fold1_test.txt"

# 数据集读取，训练集和测试集
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=TRAINING,
    target_dtype=np.int,
    features_dtype=np.float,
    target_column=-1)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=TEST,
    target_dtype=np.int,
    features_dtype=np.float,
    target_column=-1)

# 特征
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 构建DNN网络
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[256, 256],
                                            n_classes=4,
                                            dropout=0.25,
                                            model_dir="tmp/model")

# 拟合模型，迭代30000步
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=30000)

# 计算精度
accuracy_score = classifier.evaluate(x=test_set.data,y=test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))
# print(test_set.target)

# 预测新样本的类别
y_pred = list(classifier.predict(test_set.data, as_iterable=True))
maf = metrics.f1_score(test_set.target, y_pred, average='macro')  #计算macro-f-score
print('maf :',maf)
