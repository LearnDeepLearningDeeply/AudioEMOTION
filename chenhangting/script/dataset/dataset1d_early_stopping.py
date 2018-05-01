# -*- coding: utf-8 -*-
"""
@date: Created on 2018/4/23
@author: chenhangting

@notes: basic dataset module to load audio feature map

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

class AudioFeatureDataset(Dataset):
    def __init__(self,featrootdir,cvtxtrootdir,feattype,cvnum,mode,normflag,normfile):
        self.__pathfeatrootdir=featrootdir
        self.__pathcvtxt=os.path.join(cvtxtrootdir,'fold{}_{}.txt'.format(cvnum,mode))
        self.__class=('neu','hap','ang','sad')
        self.__feattype=feattype
        self.__cvnum=cvnum
        self.mode=mode
        self.normflag=normflag
        self.normfile=normfile
        if(self.normflag==1 and self.mode!='train'):warnings.warn("You want to calculate normalization coefficients according to test dataset")
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
                    self.__filedict["{}.{}".format(row[0].rsplit(".",maxsplit=1)[0],self.__feattype)]=self.__class.index(row[1])
        self.__filelist=[k for k in self.__filedict]
        self.__dim=(np.load(os.path.join(self.__pathfeatrootdir,self.__filelist[0])).shape)[1]

        self.__mean=np.zeros((self.__dim,),dtype=np.float64)
        self.__std=np.zeros((self.__dim,),dtype=np.float64)
        p=progressbar.ProgressBar()
        p.start(len(self.__filelist))
        frames=0;self.maxframes=0
        for i,filename in enumerate(self.__filelist):
            if(self.__feattype=='dat'):
                temparray=np.loadtxt(os.path.join(self.__pathfeatrootdir,filename))
            elif(self.__feattype=='npy'):
                temparray=np.load(os.path.join(self.__pathfeatrootdir,filename))
            else:
                sys.exit("Unsupported data type %s"%self.__feattype)
            if(np.any(np.isnan(temparray)) or np.any(np.isinf(temparray))):
                sys.exit("unexpected value of nan or inf in %s"%(filename,))
            frames+=temparray.shape[0]
            if(self.maxframes<temparray.shape[0]):self.maxframes=temparray.shape[0]
            self.__mean+=temparray.sum(axis=0)
            self.__std+=(temparray**2).sum(axis=0)
            p.update(i+1)
        frames=float(frames)
        self.__mean=self.__mean/frames
        self.__std=np.sqrt((self.__std/frames-self.__mean**2))
        p.finish()

        if(self.normflag==1):
            if(self.normfile[-4:]=='.npy'):
                np.save(self.normfile[:-4],np.array((self.__mean,self.__std)))
            else:
                np.save(self.normfile,np.array((self.__mean,self.__std)))
        elif(self.normflag==0):
            if(self.normfile[-4:]=='.npy'):
                temparray=np.load(self.normfile)
            else:
                temparray=np.load(self.normfile+'.npy')
            self.__mean=temparray[0]
            self.__std=temparray[1]
        elif(self.normflag==-1):
            self.__mean=np.zeros((self.__dim,),dtype=np.float64)
            self.__std=np.ones((self.__dim,),dtype=np.float64)
        else:sys.exit("Unrecognized normflag %d"%self.normflag)
        self.__mean=self.__mean.reshape((1,self.__dim))
        self.__std=self.__std.reshape((1,self.__dim))
    
    def __len__(self):
        return len(self.__filelist)
    
    def __getitem__(self,idx):
        filename=self.__filelist[idx]
        if(self.__feattype=='dat'):
            data=np.loadtxt(os.path.join(self.__pathfeatrootdir,filename))
        elif(self.__feattype=='npy'):
            data=np.load(os.path.join(self.__pathfeatrootdir,filename))
        else:
            sys.exit("Unsupported data type %s"%self.__feattype)
        mean=np.tile(self.__mean,(data.shape[0],1))
        std=np.tile(self.__std,(data.shape[0],1))
        data=(data-mean)/std
        label=[self.__filedict[filename],]*data.shape[0]

        length=data.shape[0]
        data=np.pad(data,((0,self.maxframes-length),(0,0)),'constant')
        label=np.pad(label,(0,self.maxframes-length),'constant',constant_values=(-1,-1))

        data=torch.FloatTensor(data)
        label=torch.LongTensor(label)
        return (data,label,length,os.path.basename(filename))
	
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
