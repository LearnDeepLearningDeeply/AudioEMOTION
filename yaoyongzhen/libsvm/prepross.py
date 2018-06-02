"""
@date: Created on 2018/5/23
@author: yaoyongzhen
@notes: 根据交叉验证流程的训练，验证，测试集划分，将各个集合的音频特征合并到各自单独的一个文件中
"""
import numpy as np
import pandas as pd
import os
feature_dir= 'data/'

def parse_X_y(list_file):
    file_list_file = open(list_file,'r')
    file_list = []
    label_list = []

    for line in file_list_file.readlines():
                line=line.strip('\r\n')
                line_array = line.split("\t")
                file_dir = line_array[0]
                file_list.append(file_dir)
                label_list.append(line_array[1])
    print (len(file_list),"files in total\n")

    X = pd.DataFrame()
    for i in range(len(file_list)):
        print('handle i :', i)
        file_name = file_list[i].strip(".wav")
        feature_name = feature_dir+file_name+".csv"
        print(feature_name)
        tmp = pd.read_csv(feature_name,sep=';')
        X = pd.concat([X,tmp])

    print(X.shape)
    X['label'] = label_list
    # X.insert(0,'label',label_list)
    # print(X['label'])
    del X['name']
    print(X.shape)
    X.to_csv('data/'+list_file,index=False)
    return X

def gen_cv_folds():
    rootdir = 'folds/'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        print(path)
        parse_X_y(path)

def main():
    gen_cv_folds()

if __name__ == '__main__':
    main()
