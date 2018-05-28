# -*- coding: utf-8 -*-
"""
@date: Created on 2018/4/23
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
from dataset1d_early_stopping import AudioFeatureDataset
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
parser.add_argument('--savepath',type=str,default='./model.pkl',metavar='S', \
                    help='save model in the path')


args=parser.parse_args()
os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

superParams={'input_dim':36,

            'output_dim':4,

            'dropout':0.25}
emotion_labels=('neu','hap','ang','sad')

args.cuda=torch.cuda.is_available()
if(args.cuda==False):sys.exit("GPU is not available")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)

# load dataset
dataset_train=AudioFeatureDataset(featrootdir=r'/home/liuzongming/feature_alstm_unnorm', \
                                    cvtxtrootdir='/home/liuzongming/AudioEMOTION/chenhangting/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,mode='train',normflag=1,\
                                    normfile=r'/home/liuzongming/AudioEMOTION/chenhangting/script/features/ms{}.npy'.format(args.cvnum))

dataset_eva=AudioFeatureDataset(featrootdir=r'/home/liuzongming/feature_alstm_unnorm', \
                                    cvtxtrootdir='/home/liuzongming/AudioEMOTION/chenhangting/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,mode='eva',normflag=0,\
                                    normfile=r'/home/liuzongming/AudioEMOTION/chenhangting/script/features/ms{}.npy'.format(args.cvnum))


dataset_test=AudioFeatureDataset(featrootdir=r'/home/liuzongming/feature_alstm_unnorm', \
                                    cvtxtrootdir='/home/liuzongming/AudioEMOTION/chenhangting/CV/folds',feattype='npy', \
                                    cvnum=args.cvnum,mode='test',normflag=0,\
                                    normfile=r'/home/liuzongming/AudioEMOTION/chenhangting/script/features/ms{}.npy'.format(args.cvnum))


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
    def __init__(self,input_dim,output_dim,time_steps=3,dropout=0.5):
        #dropout
        super(Net,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.time_steps = time_steps
        self.input_layer = nn.Linear(self.input_dim,256)
        self.lstm_cell = nn.LSTMCell(256,128)
        self.time_w_layer = nn.Linear(self.time_steps,1,bias=False)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        #self.w_pool = nn.Linear(hidden_dim*self.bi_num,output_dim)

        self.branch_layer = nn.Sequential(
            nn.Linear(128,4),
            nn.LogSoftmax(dim=2)
        )

    def forward(self,x,batch_size,time_steps=3):

        output_linear=self.input_layer(x)
        output_linear = self.dropout_layer(output_linear)
        h = torch.zeros(batch_size,128).cuda()
        c = torch.zeros(batch_size,128).cuda()

        h_steps = torch.zeros(batch_size,128,time_steps).cuda()
        c_steps = torch.zeros(batch_size,128,time_steps).cuda()
        out = torch.zeros(batch_size,np.shape(output_linear)[1],128).cuda()
        counter = 0
        for i in range(np.shape(output_linear)[1]):

            input_lstmcell = output_linear[:,i,:]

            h,c = self.lstm_cell(input_lstmcell,(h,c))


            h_steps[:,:,(i%time_steps)] = h
            c_steps[:,:,(i%time_steps)] = c
            counter += 1
            if (counter == time_steps):

                #print(np.shape(h_steps))
                h = torch.squeeze(self.time_w_layer(h_steps))
                #print(np.shape(h))
                #c = self.time_w_layer(c_steps)[:,:,0]
                c = torch.squeeze(self.time_w_layer(c_steps))
                h_steps = torch.zeros(batch_size,128,time_steps).cuda()
                c_steps = torch.zeros(batch_size,128,time_steps).cuda()

                counter = 0

            out[:,i,:] = h

            #h = self.dropout_layer(h)
            #c = self.dropout_layer(c)

        #out,length=pad_packed_sequence(out,batch_first=True)
        #print(out)
        out = self.dropout_layer(out)
        out = self.branch_layer(out)
        #print(out)
        return out


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

        #data=pack_padded_sequence(data,length,batch_first=True)

        optimizer.zero_grad()
        output = model(data,batch_size)
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

        #data=pack_padded_sequence(data,length,batch_first=True)

        output = model(data,batch_size)

        if torch.sum(torch.isnan(output))!=torch.tensor(0).cuda():
            print(output)
            print(np.shape(output))
            exit()

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
        test_loss/float(numframes), metrics.accuracy_score(label_true,label_pred,normalize=False), \
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

eva_fscore_list=[]
#eva_acc,eva_fscore=test(eva_loader)
for epoch in range(1,args.epoch+1):
    train(epoch,train_loader)
    eva_acc,eva_fscore=test(eva_loader)
    eva_fscore_list.append(eva_fscore)
    if(early_stopping(model,args.savepath,eva_fscore_list,gap=10)):break

model.load_state_dict(torch.load(args.savepath))
model=model.cuda()
test(test_loader)
