# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/3
@author: chenhangting

@notes: a attention-lstm  for IEMOCAP
    support early stopping
    it is a sccript for pretrain attention lstm
    the reason to do this is that the directly trained attention-lstm performs very bad
    it's a stable version with bilstm splited
    here we use a new attetion described in A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
    add penalty for various attention
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
from reverse_seq import reverse_padded_sequence
from dataset1d_early_stopping_single_label import AudioFeatureDataset
import pdb
import os
from sklearn import metrics



# Training setttings
parser=argparse.ArgumentParser(description='PyTorch for audio emotion classification in IEMOCAP')
parser.add_argument('--cvnum',type=int,default=1,metavar='N', \
                    help='the num of cv set')
parser.add_argument('--batch_size',type=int,default=16,metavar='N', \
                    help='input batch size for training ( default 16 )')
parser.add_argument('--epoch',type=int,default=150,metavar='N', \
                    help='number of epochs to train ( default 150)')
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

emotion_labels=('neu','hap','ang','sad')
superParams={'input_dim':36,

            'output_dim':len(emotion_labels),
            'layer_num':2,
            'da':64,
            'biFlag':1,
            'r':4,
            'time_steps':3,
            'dropout':0.25,
}
penaltyWeight=1.0
args.cuda=torch.cuda.is_available()
if(args.cuda==False):sys.exit("GPU is not available")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)

# load dataset
featrootdir=r'/home/liuzongming/feature_alstm_unnorm'
cvtxtrootdir='/home/liuzongming/AudioEMOTION/chenhangting/CV/folds'
normfile=r'/home/liuzongming/AudioEMOTION/chenhangting/script/features/ms{}.npy'.format(args.cvnum)

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
        self.input_layer = nn.Linear(self.input_dim,256)
        self.lstm_cell = nn.LSTMCell(256,128)
        self.time_w_layer = nn.Linear(self.time_steps,1,bias=False)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.layer_num = layer_num
        #self.w_pool = nn.Linear(hidden_dim*self.bi_num,output_dim)

        self.branch_layer = nn.Sequential(
            nn.Linear(128*(self.biFlag+1)*r,self.output_dim),
            nn.LogSoftmax(dim=1)
        )

        self.simple_attention=nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear((self.biFlag+1)*128,da,bias=True),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(da,r,bias=False),
        )

    def init_attention_weight(self,batch_size,maxlength,r):
        return Variable(torch.zeros(batch_size,maxlength,r)).cuda()

    def forward(self,x,batch_size,time_steps,length,r):

        batch_size=x.size(0)
        maxlength=x.size(1)
        weight=self.init_attention_weight(batch_size,maxlength,r)
        oneMat=Variable(torch.ones(batch_size,r,r)).cuda()

        output_linear=self.input_layer(x)

        #output_linear = self.dropout_layer(output_linear)
        #print(np.shape(output_linear))
        out = output_linear
        for l in range(self.layer_num):
            #print(l)
            output_linear = out
            h = torch.zeros(batch_size,128).cuda()
            c = torch.zeros(batch_size,128).cuda()

            h_steps = torch.zeros(batch_size,128,time_steps).cuda()
            c_steps = torch.zeros(batch_size,128,time_steps).cuda()
            out1 = torch.zeros(batch_size,np.shape(output_linear)[1],128).cuda()

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

                out1[:,i,:] = h
            if(self.biFlag):
                h = torch.zeros(batch_size, 128).cuda()
                c = torch.zeros(batch_size, 128).cuda()

                h_steps = torch.zeros(batch_size, 128, time_steps).cuda()
                c_steps = torch.zeros(batch_size, 128, time_steps).cuda()
                out2 = torch.zeros(batch_size, np.shape(output_linear)[1], 128).cuda()

                counter = 0
                for i in range(np.shape(output_linear)[1]):

                    input_lstmcell = output_linear[:,(np.shape(output_linear)[1]-i-1), :]

                    h, c = self.lstm_cell(input_lstmcell, (h, c))

                    h_steps[:, :, (i % time_steps)] = h
                    c_steps[:, :, (i % time_steps)] = c
                    counter += 1
                    if (counter == time_steps):
                        # print(np.shape(h_steps))
                        h = torch.squeeze(self.time_w_layer(h_steps))
                        # print(np.shape(h))
                        # c = self.time_w_layer(c_steps)[:,:,0]
                        c = torch.squeeze(self.time_w_layer(c_steps))
                        h_steps = torch.zeros(batch_size, 128, time_steps).cuda()
                        c_steps = torch.zeros(batch_size, 128, time_steps).cuda()

                        counter = 0

                    out2[:, i, :] = h
            if(self.biFlag==0 ):
                out = out1
            if(self.biFlag==1):
                out = torch.cat((out1,out2),2)

            out = self.dropout_layer(out)

        potential=self.simple_attention(out)
        for inx,l in enumerate(length):weight[inx,0:l,:]=F.softmax(potential[inx,0:l,:],dim=0)
        weight=torch.transpose(weight,1,2)
        out_final=torch.bmm(weight,out)
        out_final=out_final.view(batch_size,-1)
        out_final=self.branch_layer(out_final)

        penalty=torch.sum(torch.sum(torch.sum(torch.pow(torch.bmm(torch.sqrt(weight),torch.transpose(torch.sqrt(weight),1,2))-oneMat,2.0),0),0),0)

        return out_final,penalty

model=Net(**superParams)
model.cuda()
optimizer=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr,weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

def train(epoch,trainLoader):
    model.train();exp_lr_scheduler.step()
    for batch_inx,(data,target,length,name) in enumerate(trainLoader):
        batch_size=data.size(0)
        max_length=torch.max(length)
        data=data[:,0:max_length,:]

        (data,target,length,name)=sort_batch(data,target,length,name)
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)

        optimizer.zero_grad()
        output,penalty=model(data,batch_size,superParams['time_steps'],length,superParams['r'])
        numframes=0
#            label=int(torch.squeeze(target[i,0]).item())
        loss=F.nll_loss(output,target,weight=emotionLabelWeight,size_average=False)+penaltyWeight*penalty
#        numframes+=length[i]
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
        with torch.no_grad():output,penalty=model(data,batch_size,superParams['time_steps'],length,superParams['r'])
        if torch.sum(torch.isnan(output))!=torch.tensor(0).cuda():
            print(output)
            print(np.shape(output))
            exit()
        test_loss=F.nll_loss(output,target,weight=emotionLabelWeight,size_average=False).item()+penaltyWeight*penalty.item()
        for i in range(batch_size):
            result=torch.squeeze(output[i,:]).cpu().data.numpy()
            test_dict1[name[i]]=result
            test_dict2[name[i]]=target.cpu().data[i]
            numframes+=length[i]
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

eva_fscore_list=[]
for epoch in range(1,args.epoch+1):
    train(epoch,train_loader)
    eva_acc,eva_fscore=test(eva_loader)
    eva_fscore_list.append(eva_fscore)
    if(early_stopping(model,args.savepath,eva_fscore_list,gap=15)):break

model.load_state_dict(torch.load(args.savepath))
model=model.cuda()
test(test_loader)
