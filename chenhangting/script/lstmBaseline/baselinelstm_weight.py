# -*- coding: utf-8 -*-
"""
@date: Created on 2018/4/20
@author: chenhangting

@notes: a lstm baseline for iemocap
    add weight to balance classes

"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import sys
sys.path.append(r'../dataset')
from dataset1d import AudioFeatureDataset
import pdb
import os
from sklearn import metrics



# Training setttings
parser=argparse.ArgumentParser(description='PyTorch for audio emotion classification in iemocap')
parser.add_argument('--cvnum',type=int,default=1,metavar='N', \
                    help='the num of cv set')
parser.add_argument('--batch_size',type=int,default=16,metavar='N', \
                    help='input batch size for training ( default 16 )')
parser.add_argument('--epoch',type=int,default=100,metavar='N', \
                    help='number of epochs to train ( default 100)')
parser.add_argument('--lr',type=float,default=0.001,metavar='LR', \
                    help='inital learning rate (default 0.001 )')
parser.add_argument('--seed',type=int,default=1,metavar='S', \
                    help='random seed ( default 1 )')
parser.add_argument('--log_interval',type=int,default=1,metavar='N', \
                    help='how many batches to wait before logging (default 10 )')
parser.add_argument('--device_id',type=int,default=0,metavar='N', \
                    help="the device id")


args=parser.parse_args()
os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

superParams={'input_dim':153,
            'hidden_dim':256,
            'output_dim':4,
            'num_layers':4,
            'biFlag':2,
            'dropout':0.25}
emotion_labels=('neu','hap','ang','sad')

args.cuda=torch.cuda.is_available()
if(args.cuda==False):sys.exit("GPU is not available")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)

# load dataset
dataset_train=AudioFeatureDataset(featrootdir=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank', \
                                    cvtxtrootdir='/mnt/c/chenhangting/Project/iemocap/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,train=True,normflag=1,\
                                    normfile=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank/ms{}.npy'.format(args.cvnum))


dataset_test=AudioFeatureDataset(featrootdir=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank', \
                                    cvtxtrootdir='/mnt/c/chenhangting/Project/iemocap/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,train=False,normflag=0,\
                                    normfile=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank/ms{}.npy'.format(args.cvnum))


print("shuffling dataset_train")
train_loader=torch.utils.data.DataLoader(dataset_train, \
                                batch_size=args.batch_size,shuffle=False, \
                                num_workers=4,pin_memory=True)


print("shuffling dataset_test")
test_loader=torch.utils.data.DataLoader(dataset_test, \
                                batch_size=args.batch_size,shuffle=False, \
                                num_workers=4,pin_memory=True)

def sort_batch(data,label,length,name):
    batch_size=data.size(0)
#    print(np.argsort(length.numpy())[::-1].copy())
    inx=torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data=data[inx]
    label=label[inx]
    length=length[inx]
    name_new=[]
    for i in list(inx.numpy()):name_new.append(name[i])
    name=name_new
    length=list(length.numpy())
    return (data,label,length,name)

class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers,biFlag,dropout=0.5):
        #dropout
        super(Net,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        if(biFlag):self.bi_num=2
        else:self.bi_num=1
        self.biFlag=biFlag

        self.layer1=nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=biFlag)
        # out = (len batch outdim) ?
        self.layer2=nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num,output_dim),
            nn.LogSoftmax(dim=2)
        )


    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim)).cuda())
    def forward(self,x,batch_size):
        hidden1=self.init_hidden(batch_size)
        out,hidden1=self.layer1(x,hidden1)
        out,length=pad_packed_sequence(out,batch_first=True)
        out=self.layer2(out)
        return out,length

model=Net(**superParams)
model.cuda()
optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.0001)

def train(epoch,trainLoader):
    model.train()
    for batch_inx,(data,target,length,name) in enumerate(trainLoader):
        batch_size=data.size(0)
        max_length=torch.max(length)
        data=data[:,0:max_length,:];target=target[:,0:max_length]

        (data,target,length,name)=sort_batch(data,target,length,name)
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        data=pack_padded_sequence(data,length,batch_first=True)

        optimizer.zero_grad()
        output,_=model(data,batch_size)
        for i in range(batch_size):
#            print(torch.squeeze(output[i,0:length[i],:]));print(torch.squeeze(target[i,0:length[i]]))
            label=int(torch.squeeze(target[i,0]).cpu().data)
            weight=(trainLoader.dataset.ingredientWeight)[emotion_labels[label]]
            if(i==0):loss=F.nll_loss(torch.squeeze(output[i,0:length[i],:]),torch.squeeze(target[i,0:length[i]]))/weight
            else:loss+=F.nll_loss(torch.squeeze(output[i,0:length[i],:]),torch.squeeze(target[i,0:length[i]]))/weight
        loss.backward()
        optimizer.step()
        if(batch_inx % args.log_interval==0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_inx * batch_size, len(trainLoader.dataset),
                100. * batch_inx / len(trainLoader), loss.data[0]))


def test(testLoader):
    model.eval()
    test_loss=0;numframes=0
    test_dict1={};test_dict2={}

    for data,target,length,name in testLoader:
        batch_size=data.size(0)
        max_length=torch.max(length)
        data=data[:,0:max_length,:];target=target[:,0:max_length]

        (data,target,length,name)=sort_batch(data,target,length,name)
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data,volatile=True),Variable(target,volatile=True)
        data=pack_padded_sequence(data,length,batch_first=True)

        output,_=model(data,batch_size)
        for i in range(batch_size):
            result=(torch.squeeze(output[i,0:length[i],:]).cpu().data.numpy()).sum(axis=0)
            test_dict1[name[i]]=result
            test_dict2[name[i]]=target.cpu().data[i][0]
            test_loss+=F.nll_loss(torch.squeeze(output[i,0:length[i],:]),torch.squeeze(target[i,0:length[i]]),size_average=False).data[0]
            numframes+=length[i]
    if(len(test_dict1)!=len(testLoader.dataset)):
        sys.exit("some test samples are missing")

    label_true=[];label_pred=[]
    for filename,result in test_dict1.items():
#        print(test_dict2[filename])
#        print(np.argmax(result)==test_dict2[filename])
        label_true.append(test_dict2[filename]);label_pred.append(np.argmax(result))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/numframes, metrics.accuracy_score(label_true,label_pred,normalize=False), \
        len(test_dict1),metrics.accuracy_score(label_true,label_pred)))
    print(metrics.confusion_matrix(label_true,label_pred))
    print("macro f-score %f"%metrics.f1_score(label_true,label_pred,average="macro"))

for epoch in range(1,args.epoch+1):
    train(epoch,train_loader)
    test(test_loader)
