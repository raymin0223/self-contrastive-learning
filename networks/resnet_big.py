"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sub_network import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False, selfcon_pos=[False,False,False], selfcon_arch='resnet', selfcon_size='same', dataset=''):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.in_channel = in_channel
        self.dataset = dataset

        self.large = False if dataset in ['cifar10', 'cifar100', 'tinyimagenet'] else True
        if not self.large:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.bn1 = nn.BatchNorm2d(64)
        if self.large:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.selfcon_pos = selfcon_pos
        self.selfcon_arch = selfcon_arch
        self.selfcon_size = selfcon_size
        self.selfcon_layer = nn.ModuleList([self._make_sub_layer(idx, pos) for idx, pos in enumerate(selfcon_pos)])
            
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_sub_layer(self, idx, pos):
        channels = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        if self.selfcon_size == 'same':
            num_blocks = self.num_blocks
        elif self.selfcon_size == 'small':
            num_blocks = [int(n/2) for n in self.num_blocks]
        elif self.selfcon_size == 'large':
            num_blocks = [int(n*2) for n in self.num_blocks]
        elif self.selfcon_size == 'fc':
            pass
        else:
            raise NotImplemented
        
        if not pos:
            return None
        else:
            if self.selfcon_size == 'fc':
                return nn.Linear(channels[idx] * self.block.expansion, channels[-1] * self.block.expansion)
            else:
                if self.selfcon_arch == 'resnet':
                    # selfcon layer do not share any parameters
                    layers = []
                    for i in range(idx+1, 4):
                        in_planes = channels[i-1] * self.block.expansion
                        layers.append(resnet_sub_layer(self.block, in_planes, channels[i], num_blocks[i], strides[i]))
                elif self.selfcon_arch == 'vgg':
                    raise NotImplemented
                elif self.selfcon_arch == 'efficientnet':
                    raise NotImplemented

                return nn.Sequential(*layers)
        
    def forward(self, x):
        sub_out = []
        
        x = F.relu(self.bn1(self.conv1(x)))
        if self.large:
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
        if self.selfcon_layer[2]:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[2](x)
                out = torch.flatten(self.avgpool(out), 1)
            else:
                out = torch.flatten(self.avgpool(x), 1)
                out = self.selfcon_layer[2](out)
            sub_out.append(out)
            
        out = self.layer4(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        return sub_out, out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class ConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, selfcon_pos=[False,False,False], selfcon_arch='resnet', selfcon_size='same', dataset=''):
        super(ConResNet, self).__init__()
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
    
    
class CEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', method='ce', num_classes=10, dim_out=128, selfcon_pos=[False,False,False], selfcon_arch='resnet', selfcon_size='same', dataset=''):
        super(CEResNet, self).__init__()
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


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
