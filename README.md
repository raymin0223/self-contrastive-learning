# Self-Contrastive Learning: An Efficient Supervised Contrastive Framework with Single-view and Sub-network

<p align="center">
  <img src=https://user-images.githubusercontent.com/50742281/203948327-155ece9a-3ce1-4997-af9b-bbf6a535f6ec.png width="500">
</p>

This repository contains the official PyTorch implementation of the following paper: 

> **Self-Contrastive Learning: An Efficient Supervised Contrastive Framework with Single-view and Sub-network**
> Sangmin Bae*, Sungnyun Kim*, Jongwoo Ko, Gihun Lee, Seungjong Noh, Se-Young Yun (KAIST AI, SK Hynix)
> https://arxiv.org/abs/2106.15499
>
> **Abstract:** *This paper proposes an efficient supervised contrastive learning framework, called Self-Contrastive (SelfCon) learning, that self-contrasts within multiple outputs from the different levels of a multi-exit network. SelfCon learning with a single-view does not require additional augmented samples, which resolves the concerns of multi-viewed batch (e.g., high computational cost and generalization error). Unlike the previous works based on the mutual information (MI) between the multi-views in unsupervised learning, we prove the MI bound for SelfCon loss in a supervised and single-viewed framework. We also empirically analyze that the success of SelfCon learning is related to the regularization effect from the single-view and sub-network. For ImageNet, SelfCon with a single-viewed batch improves accuracy by +0.3% with 67% memory and 45% time of Supervised Contrastive (SupCon) learning, and a simple ensemble of multi-exit outputs boost performance up to +1.4%.*

## Installation
We experimented with eight RTX 3090 GPUs and CUDA version of 11.3.   
Please check below requirements and install packages from `requirements.txt`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Usage
For SelfCon linear evaluation, the following commands are examples of running the code.     
Refer to `scripts/` for SupCon pretraining and 1-stage training examples.

```bash
# Pretraining on [Dataset: CIFAR-100, Model: ResNet-18]
python main_represent.py --exp_name "resnet_fc_[False,True,False]" \
    --seed 2022 \
    --method SelfCon \
    --dataset cifar100 \
    --model resnet18 \
    --selfcon_pos "[False,True,False]" \
    --selfcon_arch "resnet" \
    --selfcon_size "fc" \
    --batch_size 1024 \
    --learning_rate 0.5 \
    --temp 0.1 \
    --epochs 1000 \
    --cosine \
    --precision
```

```bash
# Finetuning on [Dataset: CIFAR-10, Model: ResNet-18]
python main_linear.py --batch_size 512 \
    --dataset cifar100 \
    --model resnet18 \
    --learning_rate 3 \
    --weight_decay 0 \
    --selfcon_pos "[False,True,False]" \
    --selfcon_arch "resnet" \
    --selfcon_size "fc" \
    --epochs 100 \
    --lr_decay_epochs '60,80' \
    --lr_decay_rate 0.1 \
    --subnet \
    --ckpt ./save/representation/SelfCon/cifar100_models/SelfCon_cifar100_resnet18_lr_0.5_multiview_False_label_True_decay_0.0001_bsz_1024_temp_0.1_seed_2022_cosine_warm_resnet_fc_[False,True,False]/last.pth
```

### Parameters for pretraining
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. default = `resnet50`. |
| `dataset`      | Dataset to use. Options:  `cifar10`, `cifar100`, `tinyimagenet`, `imagenet100`, `imagenet`. |
| `method`      | Pretraining method. Options:  `Con`, `SupCon`, `SelfCon`. |
| `lr` | Learning rate for the pretraining. default = `0.5` for the batch size of 1024. |
| `temp` | Temperature of contrastive loss function. default = `0.07`. |
| `precision` | Action argument to use 16 bit precision. default = `False`. |
| `cosine` | Action argument to use cosine annealing scheduling. default = `False`. |
| `selfcon_pos` | Position of where to attach the sub-network. default = `[False,True,False]` for ResNet architecture. |
| `selfcon_arch` | Sub-network architecture. Options: `resnet`, `vgg`, `efficientnet`, `wrn`. default = `resnet`. |
| `selfcon_size` | Block numbers of a sub-network. Options: `fc`, `small`, `same`. default = `same`. |
| `multiview` | Action argument to use multi-viwed batch. default = `False`. |
| `label` | Action argument to use label information in a contrastive loss. default = `False`. |


### Experimental Results
See our paper for more details and extensive analyses.

<p align="center">
  <img src=https://user-images.githubusercontent.com/50742281/203949180-f1badce1-9361-422e-8f2c-e4df67ed3ce6.png width="800">
</p>
<p align="center">
  <img src=https://user-images.githubusercontent.com/50742281/203949278-e8cc0571-5baf-4abe-bf98-bb9dda1b6707.png width="800">
</p>

## Reference

"""
@article{bae2021self,
  title={Self-Contrastive Learning},
  author={Bae, Sangmin and Kim, Sungnyun and Ko, Jongwoo and Lee, Gihun and Noh, Seungjong and Yun, Se-Young},
  journal={arXiv preprint arXiv:2106.15499},
  year={2021}
}
"""