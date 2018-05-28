# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/28
@author: chenhangting

@notes: a dnn for long term feature
    add weight to balance classes
    add early stopping support
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
from dataset1d_longterm import AudioFeatureDataset
import pdb
import os
from sklearn import metrics

class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.5):
        #dropout
        super(Net,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim

        self.layer1=nn.Sequential(
            nn.Linear(self.input_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.layer2=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        self.layer3=nn.Sequential(
            nn.Linear(self.hidden_dim,self.output_dim),
            nn.LogSoftmax(dim=1),
        )

    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        return out


def train(epoch,trainLoader):
    model.train()
    for batch_inx,(data,target,name) in enumerate(trainLoader):
        batch_size=data.size(0)

        data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)

        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target,weight=emotionLabelWeight,size_average=False)

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

    for data,target,name in testLoader:
        batch_size=data.size(0)
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data,volatile=True),Variable(target,volatile=True)

        output=model(data)
        test_loss=F.nll_loss(output,target,weight=emotionLabelWeight,size_average=False)
        for i in range(batch_size):
            result=torch.squeeze(output[i,:]).cpu().data.numpy()
            test_dict1[name[i]]=result
            test_dict2[name[i]]=target.cpu().data[i]
    if(len(test_dict1)!=len(testLoader.dataset)):
        sys.exit("some test samples are missing")

    label_true=[];label_pred=[]
    for filename,result in test_dict1.items():
#        print(test_dict2[filename])
#        print(np.argmax(result)==test_dict2[filename])
        label_true.append(test_dict2[filename]);label_pred.append(np.argmax(result))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss/len(testLoader.dataset), metrics.accuracy_score(label_true,label_pred,normalize=False), \
        len(test_dict1),metrics.accuracy_score(label_true,label_pred)))
    print(metrics.confusion_matrix(label_true,label_pred))
    print("macro f-score %f"%metrics.f1_score(label_true,label_pred,average="macro"))
    return metrics.accuracy_score(label_true,label_pred),metrics.f1_score(label_true,label_pred,average="macro")

def early_stopping(network,savepath,metricsInEpochs,gap=10):
    best_metric_inx=np.argmax(metricsInEpochs)
    if(best_metric_inx==len(metricsInEpochs)-1):
        torch.save(network.state_dict(),savepath)
        return False
    elif(len(metricsInEpochs)-best_metric_inx >= gap):
        return True
    else: 
        return False
    


if __name__=='__main__':
    # Training setttings
    parser=argparse.ArgumentParser(description='PyTorch for audio emotion classification in iemocap')
    parser.add_argument('--cvnum',type=int,default=1,metavar='N', \
                        help='the num of cv set')
    parser.add_argument('--batch_size',type=int,default=256,metavar='N', \
                        help='input batch size for training ( default 64 )')
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
    parser.add_argument('--savepath',type=str,default='./model.pkl',metavar='S', \
                        help='save model in the path')


    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

    superParams={'input_dim':1582,
                'hidden_dim':256,
                'output_dim':4,
                'dropout':0.25}
    emotion_labels=('neu','hap','ang','sad')

    args.cuda=torch.cuda.is_available()
    if(args.cuda==False):sys.exit("GPU is not available")
    torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)

    featrootdir=r'H:/data/data'
    cvtxtrootdir='../../CV/folds'
    normfile=r'temp/ms{}.npy'.format(args.cvnum)



    # load dataset
    dataset_train=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype='csv', \
                                        cvnum=args.cvnum,mode='train',normflag=1,\
                                        normfile=normfile)

    dataset_eva=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype='csv', \
                                        cvnum=args.cvnum,mode='eva',normflag=0,\
                                        normfile=normfile)


    dataset_test=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype='csv', \
                                        cvnum=args.cvnum,mode='test',normflag=0,\
                                        normfile=normfile)


    print("shuffling dataset_train")
    train_loader=torch.utils.data.DataLoader(dataset_train, \
                                    batch_size=args.batch_size,shuffle=False, \
                                    num_workers=4,pin_memory=True)
    print("shuffling dataset_eva")
    eva_loader=torch.utils.data.DataLoader(dataset_eva, \
                                    batch_size=args.batch_size,shuffle=False, \
                                    num_workers=4,pin_memory=True)

    print("shuffling dataset_test")
    test_loader=torch.utils.data.DataLoader(dataset_test, \
                                    batch_size=args.batch_size,shuffle=False, \
                                    num_workers=4,pin_memory=False)

    ingredientWeight=train_loader.dataset.ingredientWeight
    emotionLabelWeight=[ 1.0/ingredientWeight[k] for k in emotion_labels ]
    emotionLabelWeight=torch.FloatTensor(emotionLabelWeight).cuda()

    model=Net(**superParams)
    model.cuda()
    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.0001)

    eva_fscore_list=[]
    for epoch in range(1,args.epoch+1):
        train(epoch,train_loader)
        eva_acc,eva_fscore=test(eva_loader)
        eva_fscore_list.append(eva_fscore)
        if(early_stopping(model,args.savepath,eva_fscore_list,gap=15)):break

    model.load_state_dict(torch.load(args.savepath))
    model=model.cuda()
    test(test_loader)