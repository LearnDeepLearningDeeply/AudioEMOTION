# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/28
@author: chenhangting

@notes: a dnn and attention lstm fusion
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.optim import lr_scheduler
import numpy as np
import sys
sys.path.append(r'../dataset')
from dataset1d_master import AudioFeatureDataset
from reverse_seq import reverse_padded_sequence
import pdb
import os
from sklearn import metrics

def sort_batch(data,label,length,name,dim):
#    print(np.argsort(length.numpy())[::-1].copy())
    inx=torch.from_numpy(np.argsort(length[dim].numpy())[::-1].copy())
    data=[d[inx] for d in data]
    label=label[inx]
    length=[l[inx] for l in length]
    name_new=[]
    for i in list(inx.numpy()):name_new.append(name[i])
    name=name_new
    length=[list(l.numpy()) for l in length]
    return (data,label,length,name)

class DNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.5):
        super(DNN,self).__init__()
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
        # self.layer3=nn.Sequential(
        #     nn.Linear(self.hidden_dim,self.output_dim),
        #     nn.LogSoftmax(dim=1),
        # )

    def forward(self,x):
        x=torch.squeeze(x,1)
        batch_size=x.size(0)
        out=self.layer1(x)
        out=self.layer2(out)
        # out=self.layer3(out)
        out=out.view(batch_size,-1)
        return out

class LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers,biFlag,da,r,dropout=0.5):
        #dropout
        super(LSTM,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        self.r=r
        self.da=da
        if(biFlag):self.bi_num=2
        else:self.bi_num=1
        self.biFlag=biFlag

        self.layer1=nn.ModuleList()
        self.layer1.append(nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=0))
        if(biFlag):
                self.layer1.append(nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=0))


        # self.layer2=nn.Sequential(
        #             nn.Linear(self.hidden_dim*self.bi_num*r,self.output_dim),
        #             nn.LogSoftmax(dim=1),
        #             )

        self.simple_attention=nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.bi_num*self.hidden_dim,da,bias=True),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(da,r,bias=False),
        )

    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(1*self.num_layers,batch_size,self.hidden_dim)).cuda(),
                Variable(torch.zeros(1*self.num_layers,batch_size,self.hidden_dim)).cuda())
    
    def init_attention_weight(self,batch_size,maxlength,r):
        return Variable(torch.zeros(batch_size,maxlength,r)).cuda()
    

    def forward(self,x,length):
        batch_size=x.size(0)
        maxlength=x.size(1)
        hidden=[ self.init_hidden(batch_size) for l in range(self.bi_num)]
        weight=self.init_attention_weight(batch_size,maxlength,self.r)
        oneMat=Variable(torch.ones(batch_size,self.r,self.r)).cuda()

        out=[x,reverse_padded_sequence(x,length,batch_first=True)]
        for l in range(self.bi_num):
            out[l]=pack_padded_sequence(out[l],length,batch_first=True)
            out[l],hidden[l]=self.layer1[l](out[l],hidden[l])
            out[l],_=pad_packed_sequence(out[l],batch_first=True)
            if(l==1):out[l]=reverse_padded_sequence(out[l],length,batch_first=True)
        
        if(self.bi_num==1):out=out[0]
        else:out=torch.cat(out,2)
        potential=self.simple_attention(out)
        for inx,l in enumerate(length):weight[inx,0:l,:]=F.softmax(potential[inx,0:l,:],dim=0)
        weight=torch.transpose(weight,1,2)
        out_final=torch.bmm(weight,out)
        out_final=out_final.view(batch_size,-1)
        # out_final=self.layer2(out_final)

        penalty=torch.sum(torch.sum(torch.sum(torch.pow(torch.bmm(torch.sqrt(weight),torch.transpose(torch.sqrt(weight),1,2))-oneMat,2.0),0),0),0)
        return out_final,length,penalty

class FUSION(nn.Module):
    def __init__(self,input_dim,output_dim,dropout=0.5):
        super(FUSION,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim

        self.layer=nn.Sequential(
            nn.Linear(self.input_dim,self.output_dim),
            nn.LogSoftmax(dim=1),
        )

    def forward(self,x):
        out=torch.cat(x,1)
        out=self.layer(out)
        return out


def train(epoch,trainLoader):
    modelDNN.train();modelLSTM.train();modelFUSION.train()
    exp_lr_scheduler.step()
    for batch_inx,(data,target,length,name) in enumerate(trainLoader):
        batch_size=data[0].size(0)
        max_length=torch.max(length[0])
        data[0]=data[0][:,0:max_length,:]

        ((data1,data2),target,(length1,_),name)=sort_batch(data,target,length,name,0)
        data1,data2,target=data1.to(device),data2.to(device),target.cuda(device)

        optimizer.zero_grad()
        output1,_,penalty=modelLSTM(data1,length1)
        output2=modelDNN(data2)
        output=modelFUSION((output1,output2))
        
#            label=int(torch.squeeze(target[i,0]).item())
        loss=F.nll_loss(output,target,weight=emotionLabelWeight,size_average=False)+penaltyWeight*penalty
        loss.backward()
        
        weight_loss=0.0;grad_total=0.0;param_num=0
        for group in optimizer.param_groups:
            if(group['weight_decay']!=0):
                for p in group['params']:
                    if(p.grad is None):continue
                    w1=p.grad.data.cpu().numpy()
                    w2=p.data.cpu().numpy()
                    if(len(w1.shape)>2 or len(w1.shape)==1):w1=w1.reshape(w1.shape[0],-1)
                    if(len(w2.shape)>2 or len(w2.shape)==1):w2=w2.reshape(w2.shape[0],-1)
                    if(len(w1.shape)==1):param_num+=w1.shape[0]
                    else:param_num+=w1.shape[0]*w1.shape[1]
                    weight_loss+=group['weight_decay']*np.linalg.norm(w2,ord='fro')
                    grad_total+=np.linalg.norm(w1,ord='fro')
#        weight_loss/=float(param_num);grad_total/=float(param_num)

        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAve loss: {:.6f} and Total weight loss {:.6f} and Total grad fro-norm {:.6f}'.format(
            epoch, batch_inx * batch_size, len(trainLoader.dataset),
            100. * batch_inx*batch_size / len(trainLoader.dataset), loss.item()/len(trainLoader.dataset),weight_loss,grad_total))


def test(testLoader):
    modelDNN.eval();modelLSTM.eval();modelFUSION.eval()
    test_loss=0.0
    test_dict1={};test_dict2={}

    for data,target,length,name in testLoader:
        batch_size=data[0].size(0)
        max_length=torch.max(length[0])
        data[0]=data[0][:,0:max_length,:]

        ((data1,data2),target,(length1,_),name)=sort_batch(data,target,length,name,0)
        data1,data2,target=data1.to(device),data2.to(device),target.cuda(device)

        with torch.no_grad():
            output1,_,penalty=modelLSTM(data1,length1)
            output2=modelDNN(data2)
            output=modelFUSION((output1,output2))
        test_loss=F.nll_loss(output,target,weight=emotionLabelWeight,size_average=False).item()+penaltyWeight*penalty.item()
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

def early_stopping(network,netname,savedir,metricsInEpochs,gap=10):
    best_metric_inx=np.argmax(metricsInEpochs)
    if(best_metric_inx==len(metricsInEpochs)-1):
        for j in range(len(network)):
            torch.save(network[j].state_dict(),os.path.join(savedir,netname[j]))
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
    parser.add_argument('--batch_size',type=int,default=32,metavar='N', \
                        help='input batch size for training ( default 32 )')
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
    parser.add_argument('--savedir',type=str,default='model/',metavar='S', \
                        help='save model in the path')


    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

    emotion_labels=('neu','hap','ang','sad')
    superParamsDNN={'input_dim':1582,
                'hidden_dim':512,
                'output_dim':4,
                'dropout':0.5}
    superParamsLSTM={'input_dim':36,
            'hidden_dim':256,
            'output_dim':4,
            'num_layers':2,
            'biFlag':2,
            'da':64,
            'r':4,
            'dropout':0.25,
    }
    superParamsFUSION={
        'input_dim':superParamsLSTM['hidden_dim']*2*superParamsLSTM['r']+superParamsDNN['hidden_dim'],
        'output_dim':4,
        'dropout':0.25,
    }
    penaltyWeight=1.0

    torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)

    featrootdir=(r'/home/liuzongming/feature_alstm_unnorm',r'../../features/IS10',)
    cvtxtrootdir='../../CV/folds'
    feattype=('npy','csv',)
    normfile=(r'temp1/ms1_{}.npy'.format(args.cvnum),r'temp1/ms2_{}.npy'.format(args.cvnum),)
    longtermFlag=(0,1,)
    device=torch.device('cuda')
    netname=('DNN{}.pkl'.format(args.cvnum),'LSTM{}.pkl'.format(args.cvnum),'FUSION{}.pkl'.format(args.cvnum))

    # load dataset
    dataset_train=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype=feattype, \
                                        cvnum=args.cvnum,mode='train',normflag=1,\
                                        normfile=normfile,longtermFlag=longtermFlag)

    dataset_eva=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype=feattype, \
                                        cvnum=args.cvnum,mode='eva',normflag=0,\
                                        normfile=normfile,longtermFlag=longtermFlag)


    dataset_test=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype=feattype, \
                                        cvnum=args.cvnum,mode='test',normflag=0,\
                                        normfile=normfile,longtermFlag=longtermFlag)


    print("shuffling dataset_train")
    train_loader=torch.utils.data.DataLoader(dataset_train, \
                                    batch_size=args.batch_size,shuffle=False, \
                                    num_workers=4,pin_memory=False)
    print("shuffling dataset_eva")
    eva_loader=torch.utils.data.DataLoader(dataset_eva, \
                                    batch_size=args.batch_size,shuffle=False, \
                                    num_workers=4,pin_memory=False)

    print("shuffling dataset_test")
    test_loader=torch.utils.data.DataLoader(dataset_test, \
                                    batch_size=args.batch_size,shuffle=False, \
                                    num_workers=4,pin_memory=False)

    ingredientWeight=train_loader.dataset.ingredientWeight
    emotionLabelWeight=[ 1.0/ingredientWeight[k] for k in emotion_labels ]
    emotionLabelWeight=torch.FloatTensor(emotionLabelWeight).to(device)

    modelDNN=DNN(**superParamsDNN);modelLSTM=LSTM(**superParamsLSTM);modelFUSION=FUSION(**superParamsFUSION)
    modelDNN.to(device);modelLSTM.to(device);modelFUSION.to(device)
    optimizer=optim.Adam((*modelDNN.parameters(),*modelLSTM.parameters(),*modelFUSION.parameters(),),lr=args.lr,weight_decay=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    eva_fscore_list=[]
    for epoch in range(1,args.epoch+1):
        train(epoch,train_loader)
        eva_acc,eva_fscore=test(eva_loader)
        eva_fscore_list.append(eva_fscore)
        if(early_stopping((modelDNN,modelLSTM,modelFUSION),netname,args.savedir,eva_fscore_list,gap=15)):break

    modelDNN.load_state_dict(torch.load(os.path.join(args.savedir,netname[0])))
    modelLSTM.load_state_dict(torch.load(os.path.join(args.savedir,netname[1])))
    modelFUSION.load_state_dict(torch.load(os.path.join(args.savedir,netname[2])))
    
    modelDNN.to(device);modelLSTM.to(device);modelFUSION.to(device)
    test(test_loader)
