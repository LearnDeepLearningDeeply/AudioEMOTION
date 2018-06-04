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
from reverse_seq import reverse_padded_sequence
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
parser.add_argument('--lr',type=float,default=0.00015,metavar='LR', \
                    help='inital learning rate (default 0.00015 )')
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
             'layer_num':1,

            'output_dim':4,
            'biFlag':1,
             'time_steps':3,
            'dropout':0.25}
emotion_labels=('neu','hap','ang','sad')

args.cuda=torch.cuda.is_available()
if(args.cuda==False):sys.exit("GPU is not available")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)


# load dataset
featrootdir=r'/media/lzm/DATA/data/feature_alstm_unnorm'
cvtxtrootdir='/media/lzm/DATA/data/Github/AudioEMOTION/chenhangting/CV/folds'
normfile=r'/media/lzm/DATA/data/Github/AudioEMOTION/chenhangting/script/features/ms{}.npy'.format(args.cvnum)

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
    def __init__(self,input_dim,output_dim,time_steps,dropout,biFlag,layer_num,da,r):
        #dropout
        super(Net,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.time_steps = time_steps
        self.biFlag = biFlag
        self.input_layer = nn.Linear(self.input_dim,512)

        self.lstm_cell = nn.LSTMCell(512,256)
        self.lstm_cell2 = nn.LSTMCell(512,256)
        self.time_w_layer = nn.Linear(self.time_steps,1,bias=False)
        self.time_w_layer2 = nn.Linear(self.time_steps, 1, bias=False)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.layer_num = layer_num
        #self.w_pool = nn.Linear(hidden_dim*self.bi_num,output_dim)

        self.branch_layer = nn.Sequential(
            nn.Linear(256*(self.biFlag+1)*r,self.output_dim),
            nn.LogSoftmax(dim=1)
        )


    def forward(self,x,batch_size,time_steps,length,r):

        batch_size=x.size(0)


        output_linear=self.input_layer(x)

        output_linear = self.dropout_layer(output_linear)
        #print(np.shape(output_linear))
        out = output_linear
        for l in range(self.layer_num):
            #print(l)
            output_linear = out
            h = torch.zeros(batch_size,256).cuda()
            c = torch.zeros(batch_size,256).cuda()

            h_steps = torch.zeros(batch_size,256,time_steps).cuda()
            c_steps = torch.zeros(batch_size,256,time_steps).cuda()
            out1 = torch.zeros(batch_size,np.shape(output_linear)[1],256).cuda()

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
                    h_steps = torch.zeros(batch_size,256,time_steps).cuda()
                    c_steps = torch.zeros(batch_size,256,time_steps).cuda()

                    counter = 0

                out1[:,i,:] = h
            if(self.biFlag):
                h = torch.zeros(batch_size, 256).cuda()
                c = torch.zeros(batch_size, 256).cuda()

                h_steps = torch.zeros(batch_size, 256, time_steps).cuda()
                c_steps = torch.zeros(batch_size, 256, time_steps).cuda()
                out2 = torch.zeros(batch_size, np.shape(output_linear)[1], 256).cuda()

                counter = 0
                for i in range(np.shape(output_linear)[1]):

                    input_lstmcell = output_linear[:,(np.shape(output_linear)[1]-i-1), :]

                    h, c = self.lstm_cell2(input_lstmcell, (h, c))

                    h_steps[:, :, (i % time_steps)] = h
                    c_steps[:, :, (i % time_steps)] = c
                    counter += 1
                    if (counter == time_steps):
                        # print(np.shape(h_steps))
                        h = torch.squeeze(self.time_w_layer2(h_steps))
                        # print(np.shape(h))
                        # c = self.time_w_layer(c_steps)[:,:,0]
                        c = torch.squeeze(self.time_w_layer2(c_steps))
                        h_steps = torch.zeros(batch_size, 256, time_steps).cuda()
                        c_steps = torch.zeros(batch_size, 256, time_steps).cuda()

                        counter = 0

                    out2[:, i, :] = h
            if(self.biFlag==0 ):
                out = out1
            if(self.biFlag==1):
                out = torch.cat((out1,out2),2)

        #print(np.shape(out))
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
