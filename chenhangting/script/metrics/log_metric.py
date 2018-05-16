#coding:utf-8
'''
@author : chenhangting
@date : 2018/4/25
@note : a script to count log files
'''

import numpy as np
import sys

num_folds=5
PATHlogroot=sys.argv[1]

acc=[];fscore=[]
for i in range(num_folds):
    fold=i+1
    PATHlog="{}{}.log".format(PATHlogroot,fold)
    with open(PATHlog,'r') as f:
        temp_acc='';temp_fscore=''
        for line in f:
            if(line[0:8]=='Test set'):temp_acc=float(line.split(" (")[1].split("%)")[0])
            elif(line[0:5]=='macro'):temp_fscore=float(line.split('f-score ')[1])
        acc.append(temp_acc)
        fscore.append(temp_fscore)

acc=np.array(acc)
fscore=np.array(fscore)
if(len(acc)==num_folds and len(fscore)==num_folds):
    print('acc:');print(acc);
    print('fscore:');print(fscore)
    print("Average acc {:.03f}-+{:.03f}".format(np.mean(acc),np.std(acc,ddof=1)))
    print("Average marco f-score {:.03f}-+{:.03f}".format(np.mean(fscore),np.std(fscore,ddof=1)))
else:
    print(acc)
    print(fscore)
