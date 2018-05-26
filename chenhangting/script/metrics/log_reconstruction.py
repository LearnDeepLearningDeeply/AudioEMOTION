#coding:utf-8
'''
@author : chenhangting
@date : 2018/5/26
@note : a script to reconstruct labels from log files to calculate metrics
'''

import numpy as np
import sys
from sklearn import metrics

def confusionMat2numpy(confusionMatS,nr,nc):
    confusionMatN=list()
    temp=''
    for s in confusionMatS:
        if(s>='0' and s<='9'):temp+=s
        else:
            if(len(temp)>0):confusionMatN.append(int(temp))
            temp=''
    confusionMatN=np.array(confusionMatN).reshape((nr,nc))
    return confusionMatN

def reconstruct(confusionMat):
    assert(len(confusionMat.shape)==2)
    nr=confusionMat.shape[0]
    nc=confusionMat.shape[1]
    groundTruth=[];pred=[]
    for r in range(nr):
        for c in range(nc):
            groundTruth=groundTruth+[r,]*confusionMat[r,c]
            pred=pred+[c,]*confusionMat[r,c]
    return groundTruth,pred


num_folds=5
PATHlogroot=sys.argv[1]
fscoreList=list();recallList=list();accList=list()
for i in range(num_folds):
    fold=i+1
    PATHlog="{}{}.log".format(PATHlogroot,fold)
    with open(PATHlog,'r') as f:
        confusionMat=''
        for line in f:
            if(line[0:2]=='[[' or line[0:2]==' ['):confusionMat+=line
            elif(line[0:5]=='macro'):continue
            else:confusionMat=''
        print("fold{}".format(fold))
        print(confusionMat)
        confusionMat=confusionMat2numpy(confusionMat,4,4)
        groundTruth,pred=reconstruct(confusionMat)
        fscoreList.append(metrics.f1_score(groundTruth,pred,average='macro'))
        recallList.append(metrics.recall_score(groundTruth,pred,average='macro'))
        accList.append(metrics.accuracy_score(groundTruth,pred,normalize=True))

print('fscore');print(fscoreList);print("%.2f"%(100*np.mean(fscoreList)))
print('recall');print(recallList);print("%.2f"%(100*np.mean(recallList)))
print('acc');print(accList);print("%.2f"%(100*np.mean(accList)))
