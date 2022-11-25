import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .sub_network import *

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv7x7(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, selfcon_pos=[False,False], selfcon_arch='wrn', selfcon_size='same', dataset=''):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        self.dropout_rate = dropout_rate
        self.num_blocks = n

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.nStages = nStages

        if dataset in ['imagenet', 'imagenet100']:
            self.conv1 = conv7x7(7,nStages[0])
        else:
            self.conv1 = conv3x3(3,nStages[0])
        if 'imagenet' in dataset:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.selfcon_pos = selfcon_pos
        self.selfcon_arch = selfcon_arch
        self.selfcon_size = selfcon_size
        self.selfcon_layer = nn.ModuleList([self._make_sub_layer(idx, pos) for idx, pos in enumerate(selfcon_pos)])
        self.dataset = dataset

        for m in self.modules():
            conv_init(m)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _make_sub_layer(self, idx, pos):
        channels = [128, 256, 512] 
        strides = [1, 2, 2]
        num_blocks = [self.num_blocks]*3
        if self.selfcon_size == 'same':
            num_blocks = num_blocks
        elif self.selfcon_size == 'small':
            num_blocks = [int((n+1)/2) for n in num_blocks]
        elif self.selfcon_size == 'large':
            num_blocks = [int(n*2) for n in num_blocks]
        elif self.selfcon_size == 'fc':
            pass
        else:
            raise NotImplemented
        
        if not pos:
            return None
        else:
            if self.selfcon_size == 'fc':
                return nn.Linear(channels[idx], channels[-1])
            else:
                if self.selfcon_arch == 'resnet':
                    raise NotImplemented
                elif self.selfcon_arch == 'vgg':
                    raise NotImplemented
                elif self.selfcon_arch == 'efficientnet':
                    raise NotImplemented
                elif self.selfcon_arch == 'wrn':
                    layers = []
                    for i in range(idx+1, 3):
                        in_planes = channels[i-1]
                        layers.append(wrn_sub_layer(wide_basic, in_planes, channels[i], num_blocks[i], self.dropout_rate, strides[i]))

                return nn.Sequential(*layers)
    
    def forward(self, x):
        sub_out = []

        x = self.conv1(x)
        # maxpool -> last map before avgpool is 4x4
        if 'imagenet' in self.dataset:
            x = self.maxpool(x)

        x = self.layer1(x)
        if self.selfcon_layer[0]:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[0](x)
                out = torch.flatten(self.avgpool(out), 1)
            else:
                out = torch.flatten(self.avgpool(x), 1)
                out = self.selfcon_layer[0](out)
            sub_out.append(out)
 
        x = self.layer2(x)
        if self.selfcon_layer[1]:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[1](x)
                out = torch.flatten(self.avgpool(out), 1)
            else:
                out = torch.flatten(self.avgpool(x), 1)
                out = self.selfcon_layer[1](out)
            sub_out.append(out)

        x = self.layer3(x)
        x = F.relu(self.bn1(x))
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = torch.flatten(x ,1)
        # out = self.linear(out)

        return sub_out, x


def wrn_16_8(**kwargs):
    return Wide_ResNet(16, 8, 0.3, **kwargs)

model_dict = {
    'wrn_16_8': [wrn_16_8, 512],
}

class ConWRN(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='wrn_16_8', head='mlp', feat_dim=128, selfcon_pos=[False,False], selfcon_arch='wrn', selfcon_size='same', dataset=''):
        super(ConWRN, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(selfcon_pos=selfcon_pos, selfcon_arch=selfcon_arch, selfcon_size=selfcon_size, dataset=dataset)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
            
            self.sub_heads = []
            for pos in selfcon_pos:
                if pos:
                    self.sub_heads.append(nn.Linear(dim_in, feat_dim))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            
            heads = []
            for pos in selfcon_pos:
                if pos:
                    heads.append(nn.Sequential(
                        nn.Linear(dim_in, dim_in),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim_in, feat_dim)
                    ))
            self.sub_heads = nn.ModuleList(heads)
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        sub_feat, feat = self.encoder(x)
        
        sh_feat = []
        for sf, sub_head in zip(sub_feat, self.sub_heads):
            sh_feat.append(F.normalize(sub_head(sf), dim=1))
        
        feat = F.normalize(self.head(feat), dim=1)
        return sh_feat, feat
    
    
class CEWRN(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='wrn_16_8', method='ce', num_classes=10, dim_out=128, selfcon_pos=[False,False], selfcon_arch='wrn', selfcon_size='same', dataset=''):
        super(CEWRN, self).__init__()
        self.method = method
        
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(selfcon_pos=selfcon_pos, selfcon_arch=selfcon_arch, selfcon_size=selfcon_size, dataset=dataset)
        
        logit_fcs, feat_fcs = [], []
        for pos in selfcon_pos:
            if pos:
                logit_fcs.append(nn.Linear(dim_in, num_classes))
                feat_fcs.append(nn.Linear(dim_in, dim_out))
                
        self.logit_fc = nn.ModuleList(logit_fcs)
        self.l_fc = nn.Linear(dim_in, num_classes)

        if method not in ['ce', 'subnet_ce', 'kd']:
            self.feat_fc = nn.ModuleList(feat_fcs)
            self.f_fc = nn.Linear(dim_in, dim_out)
            for param in self.f_fc.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        sub_feat, feat = self.encoder(x)
        
        feats, logits = [], []
        
        for idx, sh_feat in enumerate(sub_feat):
            logits.append(self.logit_fc[idx](sh_feat))
            if self.method not in ['ce', 'subnet_ce', 'kd']:
                out = self.feat_fc[idx](sh_feat)
                feats.append(F.normalize(out, dim=1))
            
        if self.method not in ['ce', 'subnet_ce', 'kd']:
            return [feats, F.normalize(self.f_fc(feat), dim=1)], [logits, self.l_fc(feat)]
        else:
            return [logits, self.l_fc(feat)]
    

class LinearClassifier_WRN(nn.Module):
    """Linear classifier"""
    def __init__(self, name='wrn_16_8', num_classes=10):
        super(LinearClassifier_WRN, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)   


if __name__ == '__main__':
    net=Wide_ResNet(16, 8, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
