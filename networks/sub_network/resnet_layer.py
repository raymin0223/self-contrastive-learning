import torch
import torch.nn as nn

__all__ = ['resnet_sub_layer', 'wrn_sub_layer']


def resnet_sub_layer(block, in_planes, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for i in range(num_blocks):
        stride = strides[i]
        layers.append(block(in_planes, planes, stride))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers)

def wrn_sub_layer(block, in_planes, planes, num_blocks, dropout_rate, stride):
    strides = [stride] + [1]*(int(num_blocks)-1)
    layers = []

    for stride in strides:
        layers.append(block(in_planes, planes, dropout_rate, stride))
        in_planes = planes

    return nn.Sequential(*layers)
