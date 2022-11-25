from __future__ import print_function

import os
import sys
import argparse
import time
import math
import copy
import random
import builtins
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from RandAugment import RandAugment

from losses import ConLoss
from utils.util import *
from utils.tinyimagenet import TinyImageNet
from utils.imagenet import ImageNetSubset
from networks.resnet_big import ConResNet
from networks.vgg_big import ConVGG
from networks.wrn_big import ConWRN
from networks.efficient_big import ConEfficientNet


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./save/representation')
    parser.add_argument('--resume', help='path of model checkpoint to resume', type=str, 
                        default='')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet', 'imagenet', 'imagenet100'])
    parser.add_argument('--data_folder', type=str, default='datasets/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)

    # model
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--selfcon_pos', type=str, default='[False,False,False]', 
                        help='where to augment the paths')
    parser.add_argument('--selfcon_arch', type=str, default='resnet', 
                        choices=['resnet', 'vgg', 'efficientnet', 'wrn'], help='which architecture to form a sub-network')
    parser.add_argument('--selfcon_size', type=str, default='same', 
                        choices=['fc', 'same', 'small'], help='argument for num_blocks of a sub-network')
    parser.add_argument('--feat_dim', type=int, default=128, 
                        help='feature dimension for mlp')

    # optimization
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--precision', action='store_true', 
                        help='whether to use 16 bit precision or not')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # important arguments
    parser.add_argument('--method', type=str, 
                        choices=['Con', 'SupCon', 'SelfCon'], help='choose method')
    parser.add_argument('--multiview', action='store_true', 
                        help='use multiview batch or not')
    parser.add_argument('--label', action='store_false',
                        help='whether to use label information or not')
    parser.add_argument('--alpha', type=float, default=0.0, 
                        help='weight for selfcon with multiview loss function')

    # other arguments
    parser.add_argument('--randaug', action='store_true', 
                        help='whether to add randaugment or not')
    parser.add_argument('--weakaug', action='store_true', 
                        help='whether to use weak augmentation or not')

    opt = parser.parse_args()

    if opt.model.startswith('vgg'):
        if opt.selfcon_pos == '[False,False,False]':
            opt.selfcon_pos = '[False,False,False,False]'
        opt.selfcon_arch = 'vgg'
    elif opt.model.startswith('wrn'):
        if opt.selfcon_pos == '[False,False,False]':
            opt.selfcon_pos = '[False,False]'
        opt.selfcon_arch = 'wrn'
                
    # set the path according to the environment
    opt.model_path = '%s/%s/%s_models' % (opt.save_dir, opt.method, opt.dataset)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'tinyimagenet':
        opt.n_cls = 200
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset == 'imagenet100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
        
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.model_name = '{}_{}_{}_lr_{}_multiview_{}_label_{}_decay_{}_bsz_{}_temp_{}_seed_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.multiview, opt.label, opt.weight_decay, opt.batch_size, 
               opt.temp, opt.seed)

    # warm-up for large-batch training,
    if opt.batch_size >= 1024:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
            
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)            
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
    if opt.exp_name:
        opt.model_name = '{}_{}'.format(opt.model_name, opt.exp_name)
        
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32
    elif opt.dataset == 'tinyimagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 64
    elif opt.dataset == 'imagenet' or opt.dataset == 'imagenet100':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = [transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)), transforms.RandomHorizontalFlip()]
    if not opt.weakaug:
        transform += [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                      transforms.RandomGrayscale(p=0.2)]
    
    transform += [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(transform)

    if opt.randaug:
        train_transform.transforms.insert(0, RandAugment(2, 9))
    if opt.multiview:
        train_transform = TwoCropTransform(train_transform)
        
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
    elif opt.dataset == 'tinyimagenet':
        train_dataset = TinyImageNet(root=opt.data_folder,
                                     transform=train_transform,
                                     download=True)
    elif opt.dataset == 'imagenet':
        traindir = os.path.join(opt.data_folder, 'train')
        train_dataset = datasets.ImageFolder(root=traindir,
                                     transform=train_transform)
    elif opt.dataset == 'imagenet100':
        traindir = os.path.join(opt.data_folder, 'train')
        train_dataset = ImageNetSubset('./utils/imagenet100.txt',
                                       root=traindir,
                                       transform=train_transform)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    return train_loader


def set_model(opt):
    model_kwargs = {'name': opt.model, 
                    'dataset': opt.dataset,
                    'selfcon_pos': eval(opt.selfcon_pos),
                    'selfcon_arch': opt.selfcon_arch,
                    'selfcon_size': opt.selfcon_size
                    }
    if opt.model.startswith('resnet'):
        model = ConResNet(**model_kwargs)
    elif opt.model.startswith('vgg'):
        model = ConVGG(**model_kwargs)
    elif opt.model.startswith('wrn'):
        model = ConWRN(**model_kwargs)
    elif opt.model.startswith('eff'):
        model = ConEfficientNet(**model_kwargs)
        
    criterion = ConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        
    return model, criterion, opt


def _train(images, labels, model, criterion, epoch, bsz, opt):
    # compute loss
    features = model(images)
    if opt.method == 'Con':
        f1, f2 = torch.split(features[1], [bsz, bsz], dim=0)
    elif opt.method == 'SupCon':
        if opt.multiview:
            f1, f2 = torch.split(features[1], [bsz, bsz], dim=0)
    else:   # opt.method == 'SelfCon'
        f1, f2 = features

    if opt.method == 'SupCon':
        # SupCon
        if opt.multiview:
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)
        # SupCon-S
        else:
            features = features[1].unsqueeze(1)
            loss = criterion(features, labels, supcon_s=True)

    elif opt.method == 'Con':
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)
    elif opt.method == 'SelfCon':
        loss = torch.tensor([0.0]).cuda()
        # SelfCon
        if not opt.multiview:
            if not opt.alpha:
                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                # SelfCon-SU
                if not opt.label:
                    loss += criterion(features)
                # SelfCon
                else:
                    loss += criterion(features, labels)
            else:
                features = f2.unsqueeze(1)
                if opt.label:
                    loss += criterion(features, labels, supcon_s=True)

                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                # SelfCon-SU*
                if not opt.label:
                    loss += opt.alpha * criterion(features, selfcon_s_FG=True) 
                # SelfCon-S*
                else:
                    loss += opt.alpha * criterion(features, labels, selfcon_s_FG=True)
        # SelfCon-M
        else: 
            if not opt.alpha:
                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                labels_repeat = torch.cat([labels, labels], dim=0)
                # SelfCon-MU
                if not opt.label:
                    loss += criterion(features) 
                # SelfCon-M
                else:
                    loss += criterion(features, labels_repeat)
            else:
                f2_1, f2_2 = torch.split(f2, [bsz, bsz], dim=0)
                features = torch.cat([f2_1.unsqueeze(1), f2_2.unsqueeze(1)], dim=1)
                # contrastive loss between F (backbone)
                if not opt.label:
                    loss += criterion(features)
                else:
                    loss += criterion(features, labels)

                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                # SelfCon-MU*
                if not opt.label:
                    loss += opt.alpha * criterion(features, selfcon_m_FG=True)
                # SelfCon-M* 
                else:
                    loss += opt.alpha * criterion(features, labels, selfcon_m_FG=True)
    else:
        raise ValueError('contrastive method not supported: {}'.
                         format(opt.method))
        
    return loss
    

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    if opt.precision:
        scaler = torch.cuda.amp.GradScaler()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = labels.shape[0]
        
        if opt.multiview:
            images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        if opt.precision:
            with torch.cuda.amp.autocast():
                loss = _train(images, labels, model, criterion, epoch, bsz, opt)
        else:
            loss = _train(images, labels, model, criterion, epoch, bsz, opt)
            
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        if not opt.precision:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

    
def main():
    opt = parse_option()
    
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
#     cudnn.deterministic = True
    
    # build model and criterion
    model, criterion, opt = set_model(opt)
    
    # build data loader
    train_loader = set_loader(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            opt.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        opt.start_epoch = 1
            
    # training routine
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if opt.save_freq:
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                        opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, epoch, save_file)


if __name__ == '__main__':
    main()
