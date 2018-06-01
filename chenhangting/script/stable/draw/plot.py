# -*- coding: utf-8 -*-
"""
Created on Mon May 21 21:24:36 2018

@author: chen hangting
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import sys

p=0.6
q=1.0

filename=sys.argv[1]
PATHWAV='./wav/{}.dat'.format(filename)
PATHWEIGHT='./weight/{}.npy'.format(filename)
PATHFIG='./fig/{}.jpg'.format(filename)

w=np.loadtxt(PATHWAV)

weight=np.load(PATHWEIGHT)
weightArgmin=np.argmin(weight[0,:])
if(weight[0,weightArgmin]==0.0):weight=weight[:,0:weightArgmin]


t_w=np.linspace(0,10,w.shape[0])
t_weight=np.linspace(0,10,weight.shape[1])
end_w=int(q*w.shape[0])
start_w=int(p*w.shape[0])
end_weight=int(q*weight.shape[1])
start_weight=int(p*weight.shape[1])


fig,axes=plt.subplots(weight.shape[0]+1)
fig.set_size_inches(10, 9)

for idx,ax in enumerate(axes):
    if(idx==0):
        ax.plot(t_w[start_w:end_w],np.abs(w)[start_w:end_w],'-')
        ax.set(title='Attention weight VS Origin signal',xlabel='time(s)',ylabel='Signal Amplitude')
        ax.set_xlim(10*p,10*q)
    else:
        ax.plot(t_weight[start_weight:end_weight],weight[idx-1,start_weight:end_weight],'-')
        ax.set(xlabel='time(s)',ylabel='weight')
        ax.set_xlim(10*p,10*q)

plt.savefig(PATHFIG)