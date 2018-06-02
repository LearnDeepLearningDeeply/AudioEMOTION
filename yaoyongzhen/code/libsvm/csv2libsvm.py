# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/23
@author: yaoyongzhen
@notes: 把csv格式转化为libsvm能处理的格式
"""
import pandas as pd
import time

def df2ffm(df, fp):
        now = time.time()
        print('Format Converting begin in time:...',now)
        columns = df.columns.values
        d = len(columns)
        feature_index = [i for i in range(d)]
        field_index = [0]*d
        field = []
        for col in columns:
            field.append(col.split('_')[0])
        index = -1
        for i in range(d):
            if i==0 or field[i]!=field[i-1]:
                index+=1
            field_index[i] = index


        with open(fp, 'w') as f:
            for row in df.values:
                line =str(row[-1])
                if line == 'neu':
                    line = str(1)
                elif line == 'hap':
                    line = str(2)
                elif line == 'ang':
                    line = str(3)
                elif line == 'sad':
                    line = str(4)
                else :
                    line = str(5)
                for i in range(0, len(row)-1):
                    if row[i]!=0:
                        line += " %d:%d" % (feature_index[i], row[i])
                line+='\n'
                f.write(line)
        print('finish convert,the cost time is ',time.time()-now)
        print('[Done]')
        print()

def main():
    # fold1
    df = pd.read_csv('data/folds/fold1_test.txt')
    fp = 'data/folds/fold1_test'
    df2ffm(df,fp)
    df = pd.read_csv('data/folds/fold1_train.txt')
    fp = 'data/folds/fold1_train'
    df2ffm(df,fp)
    # fold2
    df = pd.read_csv('data/folds/fold2_test.txt')
    fp = 'data/folds/fold2_test'
    df2ffm(df,fp)
    df = pd.read_csv('data/folds/fold2_train.txt')
    fp = 'data/folds/fold2_train'
    df2ffm(df,fp)
    # fold3
    df = pd.read_csv('data/folds/fold3_test.txt')
    fp = 'data/folds/fold3_test'
    df2ffm(df,fp)
    df = pd.read_csv('data/folds/fold3_train.txt')
    fp = 'data/folds/fold3_train'
    df2ffm(df,fp)
    # fold4
    df = pd.read_csv('data/folds/fold4_test.txt')
    fp = 'data/folds/fold4_test'
    df2ffm(df,fp)
    df = pd.read_csv('data/folds/fold4_train.txt')
    fp = 'data/folds/fold4_train'
    df2ffm(df,fp)
    # fold5
    df = pd.read_csv('data/folds/fold5_test.txt')
    fp = 'data/folds/fold5_test'
    df2ffm(df,fp)
    df = pd.read_csv('data/folds/fold5_train.txt')
    fp = 'data/folds/fold5_train'
    df2ffm(df,fp)


if __name__ == '__main__':
    main()
