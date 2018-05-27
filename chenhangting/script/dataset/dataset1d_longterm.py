# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/27
@author: chenhangting

@notes: basic dataset module to load audio feature map
        long-term feature that is one vector for one audio

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
    def __init__(self,featrootdir,cvtxtrootdir,feattype,cvnum,mode,normflag,normfile):
        self.__pathfeatrootdir=featrootdir
        self.__pathcvtxt=os.path.join(cvtxtrootdir,'fold{}_{}.txt'.format(cvnum,mode))
        self.__class=('neu','hap','ang','sad')
        self.__feattype=feattype
        self.__cvnum=cvnum
        self.mode=mode
        self.normflag=normflag
        self.normfile=normfile
        if(self.normflag==1 and self.mode!='train'):warnings.warn("You want to calculate normalization coefficients according to evaluation or test dataset")
        self.__filedict={};self.__ingredient={};self.__filelist=[]
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
                    self.__filelist.append("{}.{}".format(row[0].rsplit(".",maxsplit=1)[0],self.__feattype))
        self.__dim=(npload(os.path.join(self.__pathfeatrootdir,self.__filelist[0]),feattype).shape)[0]

        self.__mean=np.zeros((self.__dim,),dtype=np.float64)
        self.__std=np.zeros((self.__dim,),dtype=np.float64)
        p=progressbar.ProgressBar()
        p.start(len(self.__filelist))
        for i,filename in enumerate(self.__filelist):
            temparray=npload(os.path.join(self.__pathfeatrootdir,filename),feattype)
            if(np.any(np.isnan(temparray)) or np.any(np.isinf(temparray))):
                sys.exit("unexpected value of nan or inf in %s"%(filename,))
            self.__mean+=temparray
            self.__std+=temparray**2
            p.update(i+1)
        self.__mean=self.__mean/len(self.__filelist)
        self.__std=np.sqrt((self.__std/len(self.__filelist)-self.__mean**2))
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
        self.__mean=self.__mean.reshape((self.__dim,))
        self.__std=self.__std.reshape((self.__dim,))
    
    def __len__(self):
        return len(self.__filelist)
    
    def __getitem__(self,idx):
        filename=self.__filelist[idx]
        data=npload(os.path.join(self.__pathfeatrootdir,filename),self.__feattype)

        data=(data-self.__mean)/self.__std
        label=self.__filedict[filename]

        data=torch.FloatTensor(data)
        return (data,label,os.path.basename(filename))
	
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
