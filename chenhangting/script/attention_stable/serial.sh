#!/bin/bash

set -e

name=lstm_attention2_pretrain_fix_penalty
PATHPY=./${name}.py
PATHLOGDIR=../../log/attention_stable/${name}
PATHMODELLOADDIR=../../temp/attention_stable/lstm_attention
PATHMODELSAVEDIR=../../temp/attention_stable/${name}
card=5
startfold=1
foldnum=5
batch_size=64


mkdir -p $PATHLOGDIR
mkdir -p $PATHMODELSAVEDIR

for i in $(seq 0 `expr $foldnum - 1`);do
    fold=`expr $startfold + $i`
    nohup python -u $PATHPY --cvnum $fold --batch_size $batch_size --device_id $card --loadpath $PATHMODELLOADDIR/model${fold}.pkl --savepath $PATHMODELSAVEDIR/model${fold}.pkl > $PATHLOGDIR/log${fold}.log 2>&1 
done
