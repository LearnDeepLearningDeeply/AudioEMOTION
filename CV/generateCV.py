import os
import sys

PATHROOT=r'H:/2/IEMOCAP_full_release/IEMOCAP_full_release'
PATHCVDIR=r'H:/2/CV/folds'
FILTER='Ses'
labels=('neu','hap','ang','sad',)

# filetree={};file_num=0
# for i in range(5):
#     session_num=i+1
#     pathwavdir=os.path.join(PATHROOT,'Session{}\sentences\wav'.format(session_num,))
#     pathwav=os.listdir(pathwavdir)
#     pathwav=[f for f in pathwav if f[0:3]==FILTER]
#     pathwavdict={}
#     for p in pathwav:
#         _=os.listdir(os.path.join(pathwavdir,p))
#         pathwavdict[p]=[f for f in _ if f[0:3]==FILTER]
#         file_num+=len(pathwavdict[p])
#     filetree[i]=pathwavdict

# print(filetree)
# print(file_num)

filedict={};num_total1=0;num_total2=0
for i0 in range(5):
    PATHTXTDIR=os.path.join(PATHROOT,'Session{}/dialog/EmoEvaluation'.format(i0+1))
    txtlist=os.listdir(PATHTXTDIR)
    txtlist=[f for f in txtlist if f[0:3]==FILTER]
    filedict_={}
    for i1 in txtlist:
        PATHTXT=os.path.join(PATHTXTDIR,i1)
        with open(PATHTXT,'r') as f:
            for line in f:
                line_=line.strip()
                if(len(line_)>2 and line_[0]=='[' and line_[-1]==']'):
                    lineSeg=line_.split('\t')
                    filename=lineSeg[1]
                    filelabel=lineSeg[2]
                    num_total2+=1
                    if(filelabel in labels):filedict_[filename]=filelabel;num_total1+=1
    filedict[i0+1]=filedict_

print(num_total1)
print(num_total2)

for i0 in range(5):
    PATHTEST=os.path.join(PATHCVDIR,'fold{}_test.txt'.format(i0+1))
    with open(PATHTEST,'w') as f:
        for k in filedict[i0+1]:
            filename='/'.join(('Session{}'.format(i0+1),'sentences/wav',k.rsplit('_',maxsplit=1)[0],k+'.wav'))
            f.write("{}\t{}\n".format(filename,filedict[i0+1][k]))
    PATHTRAIN=os.path.join(PATHCVDIR,'fold{}_train.txt'.format(i0+1))
    with open(PATHTRAIN,'w') as f:
        for i1 in range(5):
            if(i1==i0):continue
            for k in filedict[i1+1]:
                filename='/'.join(('Session{}'.format(i1+1),'sentences/wav',k.rsplit('_',maxsplit=1)[0],k+'.wav'))
                f.write("{}\t{}\n".format(filename,filedict[i1+1][k]))