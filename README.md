# Self-Contrastive Learning: Single-viewed Supervised Contrastive Framework using Sub-network

<p align="center">
  <img src=https://user-images.githubusercontent.com/50742281/203948327-155ece9a-3ce1-4997-af9b-bbf6a535f6ec.png width="500">
</p>

This repository contains the official PyTorch implementation of the following paper: 

> **Self-Contrastive Learning: Single-viewed Supervised Contrastive Framework using Sub-network** by
> Sangmin Bae*, Sungnyun Kim*, Jongwoo Ko, Gihun Lee, Seungjong Noh, Se-Young Yun, [AAAI 2023](https://aaai.org/Conferences/AAAI-23/).
> 
> **Paper**: https://arxiv.org/abs/2106.15499
>
> **Abstract:** *Contrastive loss has significantly improved performance in supervised classification tasks by using a multi-viewed framework that leverages augmentation and label information. The augmentation enables contrast with another view of a single image but enlarges training time and memory usage. To exploit the strength of multi-views while avoiding the high computation cost, we introduce a multi-exit architecture that outputs multiple features of a single image in a single-viewed framework. To this end, we propose Self-Contrastive (SelfCon) learning, which self-contrasts within multiple outputs from the different levels of a single network. The multi-exit architecture efficiently replaces multi-augmented images and leverages various information from different layers of a network. We demonstrate that SelfCon learning improves the classification performance of the encoder network, and empirically analyze its advantages in terms of the single-view and the sub-network. Furthermore, we provide theoretical evidence of the performance increase based on the mutual information bound. For ImageNet classification on ResNet-50, SelfCon improves accuracy by +0.6% with 59% memory and 48% time of Supervised Contrastive learning, and a simple ensemble of multi-exit outputs boosts performance up to +1.5%.*

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
  * [Parameters for Pretraining](#parameters-for-pretraining)
  * [Experimental Results](#experimental-results)
* [License](#license)
* [Contact](#contact)

## Installation
We experimented with eight RTX 3090 GPUs and CUDA version of 11.3.   
Please check below requirements and install packages from `requirements.txt`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Usage
To pretrain the SelfCon model, the following command is an example of running `main_represent.py`.     

```bash
# Pretraining on [Dataset: CIFAR-100, Architecture: ResNet-18]
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

For linear evaluation, run `main_linear.py` with an appropriate `${SAVE_CKPT}`.   
For the above example, `${SAVE_CKPT}` is `./save/representation/SelfCon/cifar100_models/SelfCon_cifar100_resnet18_lr_0.5_multiview_False_label_True_decay_0.0001_bsz_1024_temp_0.1_seed_2022_cosine_warm_resnet_fc_[False,True,False]/last.pth`.

```bash
# Finetuning on [Dataset: CIFAR-100, Architecture: ResNet-18]
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
    --ckpt ${SAVE_CKPT}
```

Also, refer to `./scripts/` for SupCon pretraining and 1-stage training examples.    
For ImageNet experiments, change `--dataset` to `imagenet`, specify `--data_folder`, and set hyperparameters as denoted in the paper.

### Parameters for Pretraining
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Default: `resnet50`. |
| `dataset`      | Dataset to use. Options:  `cifar10`, `cifar100`, `tinyimagenet`, `imagenet100`, `imagenet`. |
| `method`      | Pretraining method. Options:  `Con`, `SupCon`, `SelfCon`. |
| `lr` | Learning rate for the pretraining. Default: `0.5` for the batch size of 1024. |
| `temp` | Temperature of contrastive loss function. Default: `0.07`. |
| `precision` | Whether to use mixed precision. Default: `False`. |
| `cosine` | Whether to use cosine annealing scheduling. Default: `False`. |
| `selfcon_pos` | Position where to attach the sub-network. Default: `[False,True,False]` for ResNet architectures. |
| `selfcon_arch` | Sub-network architecture. Options: `resnet`, `vgg`, `efficientnet`, `wrn`. Default: `resnet`. |
| `selfcon_size` | Block numbers of a sub-network. Options: `fc`, `small`, `same`. Default: `same`. |
| `multiview` | Whether to use multi-viwed batch. Default: `False`. |
| `label` | Whether to use label information in a contrastive loss. Default: `False`. |


### Experimental Results
See our paper for more details and extensive analyses.    
Here are some of our main results.

<p align="center">
  <img src=https://user-images.githubusercontent.com/50742281/203949180-f1badce1-9361-422e-8f2c-e4df67ed3ce6.png width="800">
</p>
<p align="center">
  <img src=https://user-images.githubusercontent.com/50742281/203949278-e8cc0571-5baf-4abe-bf98-bb9dda1b6707.png width="800">
</p>

## Citing This Work

If you find this repo useful for your research, please consider citing our paper:
```
@article{bae2021self,
  title={Self-Contrastive Learning: Single-viewed Supervised Contrastive Framework using Sub-network},
  author={Bae, Sangmin and Kim, Sungnyun and Ko, Jongwoo and Lee, Gihun and Noh, Seungjong and Yun, Se-Young},
  journal={arXiv preprint arXiv:2106.15499},
  year={2021}
}
```

## License
Distributed under the MIT License.

## Contact
* Sangmin Bae: bsmn0223@kaist.ac.kr
* Sungnyun Kim: ksn4397@kaist.ac.kr
