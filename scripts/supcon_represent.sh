#!/bin/bash

seed="0"
data="cifar100"
method="SupCon"
model="resnet18"
bsz="1024"
lr="0.5" 
label="True"
multiview="True"

python main_represent.py \
    --seed $seed \
    --method $method \
    --dataset $data \
    --model $model \
    --batch_size $bsz \
    --learning_rate $lr \
    --temp 0.1 \
    --epochs 1000 \
    --multiview \
    --cosine \
    --precision

python main_linear.py --batch_size 512 \
    --dataset $data \
    --model $model \
    --learning_rate 3 \
    --weight_decay 0 \
    --epochs 100 \
    --lr_decay_epochs '60,80' \
    --lr_decay_rate 0.1 \
    --ckpt ./save/representation/${method}/${data}_models/${method}_${data}_${model}_lr_${lr}_multiview_${multiview}_label_${label}_decay_0.0001_bsz_${bsz}_temp_0.1_seed_${seed}_cosine_warm/last.pth
