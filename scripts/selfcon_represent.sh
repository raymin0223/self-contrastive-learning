#!/bin/bash

seed="0"
method="SelfCon"
data="cifar100"
model="resnet18"
arch="resnet"
size="fc"
pos="[False,True,False]"
bsz="1024"
lr="0.5" 
label="True"
multiview="False"

python main_represent.py --exp_name "${arch}_${size}_${pos}" \
    --seed $seed \
    --method $method \
    --dataset $data \
    --model $model \
    --selfcon_pos $pos \
    --selfcon_arch $arch \
    --selfcon_size $size \
    --batch_size $bsz \
    --learning_rate $lr \
    --temp 0.1 \
    --epochs 1000 \
    --cosine \
    --precision
         
python main_linear.py --batch_size 512 \
    --dataset $data \
    --model $model \
    --learning_rate 3 \
    --weight_decay 0 \
    --selfcon_pos $pos \
    --selfcon_arch $arch \
    --selfcon_size $size \
    --epochs 100 \
    --lr_decay_epochs '60,80' \
    --lr_decay_rate 0.1 \
    --subnet \
    --ckpt ./save/representation/${method}/${data}_models/${method}_${data}_${model}_lr_${lr}_multiview_${multiview}_label_${label}_decay_0.0001_bsz_${bsz}_temp_0.1_seed_${seed}_cosine_warm_${arch}_${size}_${pos}/last.pth

