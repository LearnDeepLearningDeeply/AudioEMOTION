# -*- coding: utf-8 -*-
"""
@date: Created on 2018/4/23
@author: chenhangting

@notes: a dnn baseline for iemocap
    add weight to balance classes
    add early stopping support
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
sys.path.append(r'../dataset')
from dataset1d_early_stopping import AudioFeatureDataset
import pdb
import os
from sklearn import metrics



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

superParams={'input_dim':153,
            'hidden_dim':256,
            'output_dim':4,
            'dropout':0.25}
emotion_labels=('neu','hap','ang','sad')

args.cuda=torch.cuda.is_available()
if(args.cuda==False):sys.exit("GPU is not available")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)

# load dataset
dataset_train=AudioFeatureDataset(featrootdir=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank', \
                                    cvtxtrootdir='/mnt/c/chenhangting/Project/iemocap/chenhangting/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,mode='train',normflag=1,\
                                    normfile=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank/ms{}.npy'.format(args.cvnum))

dataset_eva=AudioFeatureDataset(featrootdir=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank', \
                                    cvtxtrootdir='/mnt/c/chenhangting/Project/iemocap/chenhangting/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,mode='eva',normflag=0,\
                                    normfile=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank/ms{}.npy'.format(args.cvnum))


dataset_test=AudioFeatureDataset(featrootdir=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank', \
                                    cvtxtrootdir='/mnt/c/chenhangting/Project/iemocap/chenhangting/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,mode='test',normflag=0,\
                                    normfile=r'/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank/ms{}.npy'.format(args.cvnum))


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

def resize_batch(data,label,length,name):
    data=data.numpy();label=label.numpy()
    data_new=np.array([]);label_new=np.array([])
    length=list(length.numpy())
    for inx,l in enumerate(length):
        if(inx==0):
            data_new=data[inx,0:l,:]
            label_new=label[inx,0:l]
        else:
            data_new=np.concatenate((data_new,data[inx,0:l,:]),axis=0)
            label_new=np.concatenate((label_new,label[inx,0:l]),axis=0)
    data=torch.FloatTensor(data_new)
    label=torch.LongTensor(label_new)
    return (data,label,length,name)

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
        )
        self.layer2=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.layer3=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        self.layer4=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.layer5=nn.Sequential(
            nn.Linear(self.hidden_dim,self.output_dim),
            nn.LogSoftmax(dim=1),
        )

    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        return out

ingredientWeight=train_loader.dataset.ingredientWeight
emotionLabelWeight=[ 1.0/ingredientWeight[k] for k in emotion_labels ]
emotionLabelWeight=torch.FloatTensor(emotionLabelWeight).cuda()

model=Net(**superParams)
model.cuda()
optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(epoch,trainLoader):
    model.train();exp_lr_scheduler.step()
    train_loss=0.0
    for batch_inx,(data,target,length,name) in enumerate(trainLoader):
        batch_size=data.size(0)

        (data,target,length,name)=resize_batch(data,target,length,name)
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)

        optimizer.zero_grad()
        output=model(data)
        lengthcount=0
        for i in range(batch_size):
            label=int(torch.squeeze(target[lengthcount]).cpu().data)
            weight=ingredientWeight[emotion_labels[label]]
            if(i==0):loss=F.nll_loss(torch.squeeze(output[lengthcount:lengthcount+length[i],:]),torch.squeeze(target[lengthcount:lengthcount+length[i]]),size_average=True)/weight
            else:loss+=F.nll_loss(torch.squeeze(output[lengthcount:lengthcount+length[i],:]),torch.squeeze(target[lengthcount:lengthcount+length[i]]),size_average=True)/weight
            lengthcount+=length[i]
        loss.backward()
        train_loss+=loss.item()

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
        
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAve loss: {:.6f} and Total weight loss {:.6f} and Total grad fro-norm {:.6f}'.format(epoch, batch_inx * batch_size, len(trainLoader.dataset),
        100. * batch_inx / len(trainLoader), loss.item(),weight_loss,grad_total))
    return train_loss/len(trainLoader.dataset)


def test(testLoader):
    model.eval()
    test_loss=0;numframes=0
    test_dict1={};test_dict2={}

    for data,target,length,name in testLoader:
        batch_size=data.size(0)

        (data,target,length,name)=resize_batch(data,target,length,name)
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data,volatile=True),Variable(target,volatile=True)

        with torch.no_grad():output=model(data)
        lengthcount=0
        for i in range(batch_size):
            label=int(torch.squeeze(target[lengthcount]).cpu().data)
            weight=ingredientWeight[emotion_labels[label]]
            result=(torch.squeeze(output[lengthcount:lengthcount+length[i],:]).cpu().data.numpy()).sum(axis=0)
            test_dict1[name[i]]=result
            test_dict2[name[i]]=target.cpu().data[lengthcount]
            test_loss+=F.nll_loss(torch.squeeze(output[lengthcount:lengthcount+length[i],:]),torch.squeeze(target[lengthcount:lengthcount+length[i]]),size_average=True).item()/weight
            numframes+=length[i]
            lengthcount+=length[i]
    if(len(test_dict1)!=len(testLoader.dataset)):
        sys.exit("some test samples are missing")

    label_true=[];label_pred=[]
    for filename,result in test_dict1.items():
#        print(test_dict2[filename])
#        print(np.argmax(result)==test_dict2[filename])
        label_true.append(test_dict2[filename]);label_pred.append(np.argmax(result))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/float(numframes), metrics.accuracy_score(label_true,label_pred,normalize=False), \
        len(test_dict1),metrics.accuracy_score(label_true,label_pred)))
    print(metrics.confusion_matrix(label_true,label_pred))
    print("macro f-score %f"%metrics.f1_score(label_true,label_pred,average="macro"))
#    return metrics.accuracy_score(label_true,label_pred),metrics.f1_score(label_true,label_pred,average="macro")
    return test_loss/len(testLoader.dataset)

#def early_stopping(network,savepath,metricsTrain,metricsEva,alpha=3.0):
#    if(len(metricsTrain)!=gap*len(metricsEva)):sys.exit("Error")
#    print(metricsTrain)
#    print(metricsEva)
#    pk=metricsTrain[-5:]
#    pk=1000.0*(sum(pk)/(float(len(pk))*min(pk))-1)
#    gl=100*(metricsEva[-1]/min(metricsEva)-1)
#    pq=gl/pk
#    print("PQ_{} is {}={}/{}".format(alpha,pq,gl,pk))
#    if(pq>alpha):
#        return True
#    else:
#        torch.save(network.state_dict(),savepath)
#        return False

def early_stopping(network,savepath,metricsTrain,metricsEva,alpha=3.0):
    if(len(metricsTrain)!=gap*len(metricsEva)):sys.exit("Error")
    print(metricsTrain)
    print(metricsEva)
    gl=100*(metricsEva[-1]/min(metricsEva)-1)
    if(gl>0):
        return True
    else:
        torch.save(network.state_dict(),savepath)
        return False

test_list=[];train_list=[]
for epoch in range(1,args.epoch+1):
    gap=5
    train_loss=train(epoch,train_loader)
    train_list.append(train_loss)
    if(epoch%gap==0):
        test_loss=test(eva_loader)
        test_list.append(test_loss)
        if(early_stopping(model,args.savepath,train_list,test_list)):break

model.load_state_dict(torch.load(args.savepath))
model=model.cuda()
test(test_loader)
