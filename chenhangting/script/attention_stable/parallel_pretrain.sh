#!/bin/bash

PATHPY=./lstm_attention_pretrain_unfix.py
PATHLOGDIR=../../log/attention_stable/lstm_attention_pretrain_unfix
PATHMODELSAVEDIR=../../temp/attention_stable/lstm_attention_pretrain_unfix
PATHMODELLOADDIR=../../temp/attention_stable/lstm_attention
startcard=0
startfold=1
foldnum=5
batch_size=16

mkdir -p $PATHLOGDIR
mkdir -p $PATHMODELSAVEDIR

for i in $(seq 0 `expr $foldnum - 1`);do
    card=`expr $startcard + $i`
    fold=`expr $startfold + $i`
    nohup python -u $PATHPY --cvnum $fold --batch_size $batch_size --device_id $card --loadpath $PATHMODELLOADDIR/model${fold}.pkl --savepath $PATHMODELSAVEDIR/model${fold}.pkl > $PATHLOGDIR/log${fold}.log 2>&1 &
done
