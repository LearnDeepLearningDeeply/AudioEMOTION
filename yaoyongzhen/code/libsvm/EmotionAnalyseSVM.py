"""
@date: Created on 2018/5/23
@author: yaoyongzhen
@notes: 利用libsvm进行训练
"""
from svmutil import *
from svm import *
import numpy as np
import pandas as pd
import os
from sklearn import metrics

def svm_model_train(train_path,test_path,result_path):
    print('read ',train_path)
    y_train,train = svm_read_problem(train_path) #训练数据集
    print('read ',test_path)
    y_test,test = svm_read_problem(test_path)   #预测数据集
    print('train')
    prob  = svm_problem(y_train, train)
    param = svm_parameter('-t 0 -c 1 -w2 100')
    model = svm_train(prob, param)

    print('predict')
    y_pred, p_acc, p_val = svm_predict(y_test, test, model)
    print(p_acc)

    # 保存数据
    accuracy = metrics.accuracy_score(y_test, y_pred) #计算accuracy
    print('accuracy :',accuracy)
    maf = metrics.f1_score(y_test, y_pred, average='macro')  #计算macro-f-score
    print('maf :',maf)
    sub = pd.DataFrame()
    sub['y_true'] = list(y_test)
    sub['y_pred'] = list(y_pred)
    sub.to_csv(result_path,index=False)

def main():
    svm_model_train('data/folds/fold1_train','data/folds/fold1_test','data/folds/result/fold1_libsvm.txt')
    svm_model_train('data/folds/fold2_train','data/folds/fold2_test','data/folds/result/fold2_libsvm.txt')
    svm_model_train('data/folds/fold3_train','data/folds/fold3_test','data/folds/result/fold3_libsvm.txt')
    svm_model_train('data/folds/fold4_train','data/folds/fold4_test','data/folds/result/fold4_libsvm.txt')
    svm_model_train('data/folds/fold5_train','data/folds/fold5_test','data/folds/result/fold5_libsvm.txt')
if __name__ == '__main__':
    main()
