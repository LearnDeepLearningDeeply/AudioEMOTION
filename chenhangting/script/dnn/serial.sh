#!/bin/bash

set -e

PATHPY=./baselinednn_weight.py
PATHLOGDIR=../../log/dnn/baselinednn_weight
PATHMODELDIR=../../temp/dnn/baselinednn_weight
card=0
startfold=1
foldnum=5
batch_size=256


mkdir -p $PATHLOGDIR
mkdir -p $PATHMODELDIR

for i in $(seq 0 `expr $foldnum - 1`);do
    fold=`expr $startfold + $i`
    nohup python -u $PATHPY --cvnum $fold --batch_size $batch_size --device_id $card --savepath $PATHMODELDIR/model${fold}.pkl > $PATHLOGDIR/log${fold}.log 2>&1 
done
