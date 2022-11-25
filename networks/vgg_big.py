'''
VGG in PyTorch
Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Dict, Any, cast


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        cfg: str = 'D',
        arch: str = 'vgg16_bn',
        init_weights: bool = True,
        selfcon_pos: List[bool] = [False,False,False,False],
        selfcon_arch: str = 'vgg',
        selfcon_size: str = 'small',
        dataset: str = ''
    ) -> None:
        super(VGG, self).__init__()
        features_lst, modules_lst = [], []
        for module in features.modules():
            if isinstance(module, nn.Sequential):
                continue
            modules_lst.append(module)
            if isinstance(module, nn.MaxPool2d):
                features_lst.append(modules_lst)
                modules_lst = []
        self.block1 = nn.Sequential(*features_lst[0])
        self.block2 = nn.Sequential(*features_lst[1])
        self.block3 = nn.Sequential(*features_lst[2])
        self.block4 = nn.Sequential(*features_lst[3])
        self.block5 = nn.Sequential(*features_lst[4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.arch = arch
        self.selfcon_pos = selfcon_pos
        self.selfcon_arch = selfcon_arch
        self.selfcon_size = selfcon_size
        self.dataset = dataset
        self.selfcon_layer = nn.ModuleList([self._make_sub_layer(idx, pos, cfg) for idx, pos in enumerate(selfcon_pos)])

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        sub_out = []

        x = self.block1(x)
        if self.selfcon_layer[0]:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[0](x)
                out = torch.flatten(self.avgpool(out), 1)
            else:
                out = torch.flatten(self.avgpool(x), 1)
                out = self.selfcon_layer[0](out)
            sub_out.append(out)

        x = self.block2(x)
        if self.selfcon_layer[1]:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[1](x)
                out = torch.flatten(self.avgpool(out), 1)
            else:
                out = torch.flatten(self.avgpool(x), 1)
                out = self.selfcon_layer[1](out)
            sub_out.append(out)
 
        x = self.block3(x)
        if self.selfcon_layer[2]:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[2](x)
                out = torch.flatten(self.avgpool(out), 1)
            else:
                out = torch.flatten(self.avgpool(x), 1)
                out = self.selfcon_layer[2](out)
            sub_out.append(out)

        x = self.block4(x)
        if self.selfcon_layer[3]:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[3](x)
                out = torch.flatten(self.avgpool(out), 1)
            else:
                out = torch.flatten(self.avgpool(x), 1)
                out = self.selfcon_layer[3](out)
            sub_out.append(out)

        x = self.block5(x) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return sub_out, x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_sub_layer(self, idx, pos, cfg):
        channels = [64, 128, 256, 512, 512]
        
        if not pos:
            return None
        else:
            if self.selfcon_arch == 'resnet':
                raise NotImplemented
            elif self.selfcon_arch == 'vgg':
                if self.selfcon_size == 'fc':
                    layers = [nn.Linear(channels[idx], channels[-1])]
                else:
                    layers = []
                    if self.selfcon_size == 'same':
                        num_blocks = 3 if cfg == 'D' else 2
                    elif self.selfcon_size == 'small':
                        num_blocks = 1
                    elif self.selfcon_size == 'large':
                        raise NotImplemented
                    
                    for i in range(idx+1, 5):
                        in_planes = channels[i-1]
                        v = channels[i]
                        for b in range(num_blocks):
                            if self.arch.endswith('_bn'):
                                layers += [nn.Conv2d(in_planes, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                            else:
                                layers += [nn.Conv2d(in_planes, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                            in_planes = v
                        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplemented
                
            return nn.Sequential(*layers)
         

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), cfg=cfg, arch=arch, **kwargs)
    return model

def vgg13(progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg13', 'B', False, progress, **kwargs)

def vgg13_bn(progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg13_bn', 'B', True, progress, **kwargs)

def vgg16(progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg16', 'D', False, progress, **kwargs)

def vgg16_bn(progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg16_bn', 'D', True, progress, **kwargs)

model_dict = {'vgg13': vgg13,
              'vgg13_bn': vgg13_bn,
              'vgg16': vgg16,
              'vgg16_bn': vgg16_bn
              }

class ConVGG(nn.Module):
    def __init__(self, name='vgg13_bn', head='mlp', feat_dim=128, selfcon_pos=[False,False,False,False], selfcon_arch='vgg', selfcon_size='same', dataset=''):
        super(ConVGG, self).__init__()
        model_fun = model_dict[name]
        dim_in = 512
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
    
    
class CEVGG(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='vgg13_bn', method='ce', num_classes=10, dim_out=128, selfcon_pos=[False,False,False,False], selfcon_arch='vgg', selfcon_size='same', dataset=''):
        super(CEVGG, self).__init__()
        self.method = method
        
        model_fun = model_dict[name]
        dim_in = 512
        self.encoder = model_fun(selfcon_pos=selfcon_pos, selfcon_arch=selfcon_arch, selfcon_size=selfcon_size, dataset=dataset)
        
        logit_fcs, feat_fcs = [], []
        for pos in selfcon_pos:
            if pos:
                logit_fcs.append(nn.Sequential(nn.Linear(dim_in, dim_in),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Linear(dim_in, num_classes)
                                ))
                feat_fcs.append(nn.Linear(dim_in, dim_out))
                
        self.logit_fc = nn.ModuleList(logit_fcs)
        self.l_fc = nn.Sequential(nn.Linear(dim_in, dim_in),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Linear(dim_in, num_classes)
                                )
        
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


class LinearClassifier_VGG(nn.Module):
    """Linear classifier"""
    def __init__(self, name='vgg13_bn', num_classes=10):
        super(LinearClassifier_VGG, self).__init__()
        feat_dim = 512
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(feat_dim, num_classes)
        
    def forward(self, features):
        features = self.dropout(self.relu(self.fc1(features)))
        return self.fc2(features)
