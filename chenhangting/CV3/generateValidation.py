from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler 
import csv
import os
import sys
import numpy as np

PATHRAWTRAIN=r'H:\AudioEMOTION\chenhangting\CV\folds'
PATHEARLYPATH=r'H:\AudioEMOTION\chenhangting\CV3\folds'

def writeCSV(samples,labels,PATHCSV):
    temp=[]
    for i in range(len(samples)):temp.append((samples[i],labels[i]))
    with open(PATHCSV,'w',newline='') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerows(temp)

for i in range(5):
    fold=i+1
    filedict={}
    PATHRAWTRAINTXT=os.path.join(PATHRAWTRAIN,'fold{}_train.txt'.format(fold))
    with open(PATHRAWTRAINTXT,'r',newline='') as fraw:
        reader=csv.reader(fraw,delimiter='\t')
        for row in reader:
            if(len(row)!=2):sys.exit("row len not match")
            filedict[row[0]]=row[1]
        filedict=filedict.items()
        samples=[];labels=[]
        for (sample,label) in filedict:
            samples.append(sample)
            labels.append(label)

        ros = RandomOverSampler(random_state=42)
        samples=np.array(samples).reshape(-1,1)
        samples, labels = ros.fit_sample(samples, labels)
        samples=list(samples.reshape(-1))

        writeCSV(samples,labels,os.path.join(PATHEARLYPATH,'fold{}_train.txt'.format(fold)))