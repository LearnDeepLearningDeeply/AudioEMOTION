from sklearn.model_selection import StratifiedShuffleSplit
import csv
import os
import sys

PATHRAWTRAIN=r'H:\AudioEMOTION\CV\folds'
PATHEARLYPATH=r'H:\AudioEMOTION\chenhangting\CV\folds'
split_num=10

def writeCSV(samples,labels,inx,PATHCSV):
    temp=[]
    for i in inx:temp.append((samples[i],labels[i]))
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
        sss=StratifiedShuffleSplit(n_splits=split_num, test_size=1.0/float(split_num), random_state=0)
        for train_index,test_index in sss.split(samples,labels):
            writeCSV(samples,labels,train_index,os.path.join(PATHEARLYPATH,'fold{}_train.txt'.format(fold)))
            writeCSV(samples,labels,test_index,os.path.join(PATHEARLYPATH,'fold{}_eva.txt'.format(fold)))
            break