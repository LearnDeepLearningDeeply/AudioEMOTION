# -*- coding: utf-8 -*-
"""
@date: Created on 2018/6/1
@author: chenhangting

@notes: load attention model
		output attention weight for drawing
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import sys
sys.path.append(r'../../dataset')
from reverse_seq import reverse_padded_sequence
from dataset1d_early_stopping_single_label import AudioFeatureDataset
import pdb
import os
from sklearn import metrics

def sort_batch(data,label,length,name):
    batch_size=data.size(0)
    #print(np.argsort(length.numpy())[::-1].copy())
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
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers,biFlag,da,r,dropout=0.5):
        #dropout
        super(Net,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
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


        self.layer2=nn.Sequential(
                    nn.Linear(self.hidden_dim*self.bi_num*r,self.output_dim),
                    nn.LogSoftmax(dim=1),
                    )

        self.simple_attention=nn.Sequential(
            nn.Linear(self.bi_num*self.hidden_dim,da,bias=True),
            nn.Tanh(),
            nn.Linear(da,r,bias=False),
        )

    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(1*self.num_layers,batch_size,self.hidden_dim)).cuda(),
                Variable(torch.zeros(1*self.num_layers,batch_size,self.hidden_dim)).cuda())
    
    def init_attention_weight(self,batch_size,maxlength,r):
        return Variable(torch.zeros(batch_size,maxlength,r)).cuda()
    

    def forward(self,x,length,r):
        batch_size=x.size(0)
        maxlength=x.size(1)
        hidden=[ self.init_hidden(batch_size) for l in range(self.bi_num)]
        weight=self.init_attention_weight(batch_size,maxlength,r)
        oneMat=Variable(torch.ones(batch_size,r,r)).cuda()

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
        out_final=self.layer2(out_final)

        penalty=torch.sum(torch.sum(torch.sum(torch.pow(torch.bmm(torch.sqrt(weight),torch.transpose(torch.sqrt(weight),1,2))-oneMat,2.0),0),0),0)
        return out_final,length,penalty,weight

	
def test(testLoader,savefile):
    model.eval()
    test_loss=0;numframes=0
    test_dict1={};test_dict2={}

    for data,target,length,name in testLoader:
        batch_size=data.size(0)
        max_length=torch.max(length)
        data=data[:,0:max_length,:]

        (data,target,length,name)=sort_batch(data,target,length,name)
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data,volatile=True),Variable(target,volatile=True)

        with torch.no_grad():output,_,penalty,attentionWeight=model(data,length,superParams['r'])
        test_loss=F.nll_loss(output,target,weight=emotionLabelWeight,size_average=False).item()+1.0*penalty.item()
        for i in range(batch_size):
            print(name[i])
            if(savefile==name[i]):np.save(args.savepath,attentionWeight[i,:,:].cpu().data.numpy())
            result=torch.squeeze(output[i,:]).cpu().data.numpy()
            test_dict1[name[i]]=result
            test_dict2[name[i]]=target.cpu().data[i]
            numframes+=length[i]
    if(len(test_dict1)!=len(testLoader.dataset)):
        sys.exit("some test samples are missing")

    label_true=[];label_pred=[]
    for filename,result in test_dict1.items():
        label_true.append(test_dict2[filename]);label_pred.append(np.argmax(result))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss/len(testLoader.dataset), metrics.accuracy_score(label_true,label_pred,normalize=False), \
        len(test_dict1),metrics.accuracy_score(label_true,label_pred)))
    print(metrics.confusion_matrix(label_true,label_pred))
    print("macro f-score %f"%metrics.f1_score(label_true,label_pred,average="macro"))
    return metrics.accuracy_score(label_true,label_pred),metrics.f1_score(label_true,label_pred,average="macro")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='PyTorch for audio emotion classification in IEMOCAP')
    parser.add_argument('--cvnum',type=int,default=1,metavar='N', \
                        help='the num of cv set')
    parser.add_argument('--batch_size',type=int,default=32,metavar='N', \
                        help='input batch size for training ( default 64 )')
    parser.add_argument('--seed',type=int,default=1,metavar='S', \
                        help='random seed ( default 1 )')
    parser.add_argument('--device_id',type=int,default=2,metavar='N', \
                        help="the device id")
    parser.add_argument('--loadpath',type=str,default='./model.pkl',metavar='S', \
                        help='load the model in the path')
    parser.add_argument('--savepath',type=str,default='./weight/Ses01M_script01_3_M021.npy',metavar='S', \
                        help='save atttention weight in the path')
    parser.add_argument('--filename',type=str,default='Ses01M_script01_3_M021.npy',metavar='S', \
                        help='the file to output attention weight')

    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

    emotion_labels=('neu','hap','ang','sad')
    superParams={'input_dim':36,
                'hidden_dim':256,
                'output_dim':len(emotion_labels),
                'num_layers':2,
                'biFlag':2,
                'da':64,
                'r':4,
                'dropout':0.25,
    }

    args.cuda=torch.cuda.is_available()
    if(args.cuda==False):sys.exit("GPU is not available")
    torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)

    # load dataset
    featrootdir=r'/home/liuzongming/feature_alstm_unnorm'
    cvtxtrootdir='../../../CV/folds'
    normfile=r'../temp1/ms{}.npy'.format(args.cvnum)

    dataset_train=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype='npy', \
                                        cvnum=args.cvnum,mode='train',normflag=1,\
                                        normfile=normfile)

    dataset_eva=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype='npy', \
                                        cvnum=args.cvnum,mode='eva',normflag=0,\
                                        normfile=normfile)


    dataset_test=AudioFeatureDataset(featrootdir=featrootdir, \
                                        cvtxtrootdir=cvtxtrootdir,feattype='npy', \
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


    model=Net(**superParams);model.cuda()
    model.load_state_dict(torch.load(args.loadpath));model=model.cuda()
    test(test_loader,args.filename)
