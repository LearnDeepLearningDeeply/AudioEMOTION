#!/bin/bash

set -e

PATHPY=./lstm_attention2_pretrain_fix_penalty.py
PATHLOGDIR=../../log/attention_stable/lstm_attention2_pretrain_fix_penalty
PATHMODELLOADDIR=../../temp/attention_stable/lstm_attention
PATHMODELSAVEDIR=../../temp/attention_stable/lstm_attention2_pretrain_fix_penalty
card=2
startfold=1
foldnum=5
batch_size=64


mkdir -p $PATHLOGDIR
mkdir -p $PATHMODELSAVEDIR

for i in $(seq 0 `expr $foldnum - 1`);do
    fold=`expr $startfold + $i`
    nohup python -u $PATHPY --cvnum $fold --batch_size $batch_size --device_id $card --loadpath $PATHMODELLOADDIR/model${fold}.pkl --savepath $PATHMODELSAVEDIR/model${fold}.pkl > $PATHLOGDIR/log${fold}.log 2>&1 
done
