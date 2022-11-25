#!/bin/bash

seed="0"
data="cifar100"
bsz="1024"
method="ce"
model="resnet18"
lr="0.8"

python main_ce.py \
    --seed $seed \
    --dataset $data \
    --batch_size $bsz \
    --method $method
    --model $model \
    --learning_rate $lr \
    --epochs 500 \
    --cosine
