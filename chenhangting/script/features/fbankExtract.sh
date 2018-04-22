#!/bin/bash

PATHCAT=/mnt/c/chenhangting/Project/iemocap/CV/folds/cat.txt
PATHWAV=/mnt/c/chenhangting/Datasets/IEMOCAP_full_release
PATHNPY=/mnt/c/chenhangting/Project/iemocap/chenhangting/features/basicfbank
PATHEXE=/mnt/c/chenhangting/Programs/MFCC/mfcc
PATHTEMP=./temp
PATHFILELIST=./fileList.txt
nj=5
filenum=4490

mkdir -p $PATHNPY
rm ${temp}/* 2>/dev/null
rm -f $PATHFILELIST && touch $PATHFILELIST
while read filename filelabel; do
#	echo -e "$filename\t$filelabel"
	filenpyname="${filename%.wav}.npy"
	basedir=$PATHNPY/${filename%/*}
	mkdir -p $basedir
	echo -e "$PATHWAV/$filename\t$PATHNPY/$filenpyname" >> $PATHFILELIST
done < $PATHCAT

filenumper=`expr $filenum / $nj`
echo "each job has $filenumper files to process"
split -d -l $filenumper $PATHFILELIST $PATHTEMP/file

for i in $(seq -f %02g 0 `expr $nj - 1`);do 
	echo "[Frame];
sampleRate = 16000 ;
hipassfre = 8000 ;
lowpassfre = 10 ;
preemphasise = 0.97 ;
wlen = 400 ;
inc = 160 ;
saveType = n ;
vecNum = 1 ;
fileList = ${PATHTEMP}/file${i} ;

[MFCC];
fbankFlag = 1 ;
bankNum = 40 ;
MFCCNum = -1 ;
MFCC0thFlag = 0 ;
	 
[Others];
energyFlag = 1 ;
zeroCrossingFlag = 1 ;
brightFlag = 1 ;
subBandEFlag = 8 ;
fftLength = 0 ;
	 
[Regression];
regreOrder = 3 ;
delwin = 9 ;" > $PATHTEMP/config$i.ini

nohup $PATHEXE $PATHTEMP/config$i.ini >$PATHTEMP/log$i.log 2>&1 &

done


