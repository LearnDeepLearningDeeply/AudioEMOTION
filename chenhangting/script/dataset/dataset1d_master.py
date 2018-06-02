# -*- coding: utf-8 -*-
"""
@date: Created on 2018/6/1
@author: chenhangting

@notes: versatile dataset class for short-term and long-term,
    single label and frame label
"""

import os
import torch
import numpy as np
import warnings
import csv
import sys
import time
import progressbar
from torch.utils.data import Dataset,DataLoader

def npload(filename,feattype):
    if(feattype=='csv'):
        temp=np.loadtxt(filename,dtype=np.str,delimiter=';')
        temp=temp[1,2:].astype(np.float64)
        return temp
    elif(feattype=='npy'):
        return np.load(filename)
    elif(feattype=='txt' or feattype=='dat'):
        return np.loadtxt(filename)
    else:
        sys.exit("Unsupported data type %s"%feattype)

class AudioFeatureDataset(Dataset):
    def __init__(self,featrootdir,cvtxtrootdir,feattype,cvnum,mode,normflag,normfile,longtermFlag):
        self.__pathfeatrootdir=featrootdir
        self.__pathcvtxt=os.path.join(cvtxtrootdir,'fold{}_{}.txt'.format(cvnum,mode))
        self.__class=('neu','hap','ang','sad')
        self.__feattype=feattype
        self.__cvnum=cvnum
        self.mode=mode
        self.normflag=normflag
        self.normfile=normfile
        self.longtermFlag=longtermFlag
        if(len(featrootdir)==len(feattype)==len(normfile)==len(longtermFlag)):pass
        else:sys.exit("len(featrootdir)==len(feattype)==len(normfile)==len(longtermFlag)")
        self.numFeatures=len(featrootdir)
        if(self.normflag==1 and self.mode!='train'):warnings.warn("You want to calculate normalization coefficients according to no train dataset")

        self.__filedict={};self.__ingredient={}
        for c in self.__class:self.__ingredient[c]=0
        with open(self.__pathcvtxt,'r',newline='') as f:
            reader=csv.reader(f,delimiter='\t')
            for row in reader:
                if(len(row)!=2):
                    print(row)
                    sys.exit("Wrong in cv txt file %s"%self.__pathcvtxt)
                else:
                    self.__ingredient[row[1]]+=1
                    self.__filedict[row[0].rsplit(".",maxsplit=1)[0]]=self.__class.index(row[1])
        self.__filelist=[k for k in self.__filedict]

        self.__dim=list();self.__mean=list();self.__std=list()
        for i in range(self.numFeatures):
            self.__dim.append((npload(os.path.join(self.__pathfeatrootdir[i],self.__filelist[0]+'.'+feattype[i]),feattype[i]).shape)[-1])
            self.__mean.append(np.zeros((self.__dim[i],),dtype=np.float64))
            self.__std.append(np.zeros((self.__dim[i],),dtype=np.float64))
        
        p=progressbar.ProgressBar()
        p.start(len(self.__filelist))
        frames=[1,]*self.numFeatures;self.maxframes=[1,]*self.numFeatures
        for i,filename in enumerate(self.__filelist):
            for j in range(self.numFeatures):
                temparray=npload(os.path.join(self.__pathfeatrootdir[j],filename+'.'+self.__feattype[j]),self.__feattype[j])
                if(np.any(np.isnan(temparray)) or np.any(np.isinf(temparray))):
                    sys.exit("unexpected value of nan or inf in %s"%(filename,))
                if(self.longtermFlag[j]==1):
                    frames[j]+=1
                    self.__mean[j]+=temparray
                    self.__std[j]+=temparray**2
                else:
                    frames[j]+=temparray.shape[0]
                    self.__mean[j]+=temparray.sum(axis=0)
                    self.__std[j]+=(temparray**2).sum(axis=0)
            if(self.maxframes[j]<temparray.shape[0] and self.longtermFlag[j]==0):self.maxframes[j]=temparray.shape[0]
            p.update(i+1)
        for j in range(self.numFeatures):
            self.__mean[j]=self.__mean[j]/frames[j]
            self.__std[j]=np.sqrt((self.__std[j]/frames[j]-self.__mean[j]**2))
        p.finish()

        for j in range(self.numFeatures):
            if(self.normflag==1):
                if(self.normfile[j][-4:]=='.npy'):
                    np.save(self.normfile[j][:-4],np.array((self.__mean[j],self.__std[j])))
                else:
                    np.save(self.normfile[j],np.array((self.__mean[j],self.__std[j])))
            elif(self.normflag==0):
                if(self.normfile[j][-4:]=='.npy'):
                    temparray=np.load(self.normfile[j])
                else:
                    temparray=np.load(self.normfile[j]+'.npy')
                self.__mean[j]=temparray[0]
                self.__std[j]=temparray[1]
            elif(self.normflag==-1):
                self.__mean[j]=np.zeros((self.__dim[j],),dtype=np.float64)
                self.__std[j]=np.ones((self.__dim[j],),dtype=np.float64)
            else:sys.exit("Unrecognized normflag %d"%self.normflag)
            self.__mean[j]=self.__mean[j].reshape((1,self.__dim[j]))
            self.__std[j]=self.__std[j].reshape((1,self.__dim[j]))
    
    def __len__(self):
        return len(self.__filelist)
    
    def __getitem__(self,idx):
        filename=self.__filelist[idx]
        dataList=list()
        lengthList=list()
        for j in range(self.numFeatures):
            data=npload(os.path.join(self.__pathfeatrootdir[j],filename+'.'+self.__feattype[j]),self.__feattype[j])
            length=data.shape[0]
            if(self.longtermFlag[j]==0):
                mean=np.tile(self.__mean[j],(data.shape[0],1))
                std=np.tile(self.__std[j],(data.shape[0],1))
                lengthList.append(data.shape[0])
            else:
                mean=self.__mean[j]
                std=self.__std[j]
                lengthList.append(1)
            data=(data-mean)/std
            if(self.longtermFlag[j]==0):data=np.pad(data,((0,self.maxframes[j]-length),(0,0)),'constant')
            dataList.append(torch.FloatTensor(data))

        label=self.__filedict[filename]
        return (dataList,label,lengthList,os.path.basename(filename))
	
    @property
    def ingredientCount(self):
        return self.__ingredient
	
    @property
    def ingredientWeight(self):
        ingredientWeightdict={}
        for c in self.__class:
            ingredientWeightdict[c]=float(self.__ingredient[c])/float(len(self.__filelist))
        return ingredientWeightdict
	
    @property
    def dim(self):
        return self.__dim
	
    @property
    def cvnum(self):
        return self.__cvnum
	
    @property
    def filelist(self):
        return self.__filelist
