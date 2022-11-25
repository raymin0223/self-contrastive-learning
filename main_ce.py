from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random
import builtins
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.autograd import Variable

from networks.resnet_big import CEResNet
from networks.vgg_big import CEVGG
from networks.wrn_big import CEWRN
from networks.efficient_big import CEEffNet
from losses import *
from utils.util import *
from utils.tinyimagenet import TinyImageNet
from utils.imagenet import ImageNetSubset


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
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
    parser.add_argument('--dim_out', default=128, type=int, 
                        help='feat dimension for CEResNet')

    # optimization
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.2)
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    # important arguments
    parser.add_argument('--method', type=str, 
                        choices=['ce', 'subnet_ce', 'kd', 'selfcon'], help='choose method')
    parser.add_argument('--alpha', type=float, default=0., help='weight balance for subnet CE')
    parser.add_argument('--beta', type=float, default=0., help='weight balance for KD')
    parser.add_argument('--gamma', type=float, default=0., help='weight balance for other losses')
    parser.add_argument('--temperature', type=float, default=3.0, help='temperature for KD loss function')

    opt = parser.parse_args()

    if opt.model.startswith('vgg'):
        if opt.selfcon_pos == '[False,False,False]':
            opt.selfcon_pos = '[False,False,False,False]'
        opt.selfcon_arch = 'vgg'
    if opt.model.startswith('eff'):
        if opt.selfcon_pos == '[False,False,False]':
            opt.selfcon_pos = '[False]'
        opt.selfcon_arch = 'eff'
    
    # set the path according to the environment
    opt.model_path = './save/distill/%s/%s_models' % (opt.method, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_seed_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.seed)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.exp_name:
        opt.model_name = '{}_{}'.format(opt.model_name, opt.exp_name)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.n_data = 50000
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.n_data = 50000
    elif opt.dataset == 'tinyimagenet':
        opt.n_cls = 200
        opt.n_data = 100000
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
        opt.n_data = 1200000
    elif opt.dataset == 'imagenet100':
        opt.n_cls = 100
        opt.n_data = 120000
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    if opt.method == 'ce':
        opt.alpha, opt.beta, opt.gamma = 0, 0, 0
    elif opt.method == 'subnet_ce':
        opt.alpha, opt.beta, opt.gamma = 1.0, 0, 0
    elif opt.method == 'kd':
        opt.alpha, opt.beta, opt.gamma = 0.5, 0.5, 0
    elif opt.method == 'selfcon':
        opt.alpha, opt.beta, opt.gamma = 1.0, 0, 0.8
        
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

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset not in ['imagenet', 'imagenet100']:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        val_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                            transform=train_transform,
                                            download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                            transform=train_transform,
                                            download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'tinyimagenet':
        train_dataset = TinyImageNet(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
        val_dataset = TinyImageNet(root=opt.data_folder,
                                   train=False,
                                   transform=val_transform)
    elif opt.dataset == 'imagenet':
        traindir = os.path.join(opt.data_folder, 'train')
        valdir = os.path.join(opt.data_folder, 'val')
        train_dataset = datasets.ImageFolder(root=traindir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=valdir, transform=val_transform)
    elif opt.dataset == 'imagenet100':
        traindir = os.path.join(opt.data_folder, 'train')
        valdir = os.path.join(opt.data_folder, 'val')
        
        train_dataset = ImageNetSubset('./utils/imagenet100.txt',
                                       root=traindir,
                                       transform=train_transform)
        val_dataset = ImageNetSubset('./utils/imagenet100.txt',
                                     root=valdir,
                                     transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model_kwargs = {'name': opt.model, 
                    'method': opt.method,
                    'num_classes': opt.n_cls,
                    'dim_out': opt.dim_out,
                    'dataset': opt.dataset,
                    'selfcon_pos': eval(opt.selfcon_pos),
                    'selfcon_arch': opt.selfcon_arch,
                    'selfcon_size': opt.selfcon_size
                    }

    if opt.model.startswith('resnet'):
        model = CEResNet(**model_kwargs)
    elif opt.model.startswith('vgg'):
        model = CEVGG(**model_kwargs)
    elif opt.model.startswith('wrn'):
        model = CEWRN(**model_kwargs)
    elif opt.model.startswith('eff'):
        model = CEEffNet(**model_kwargs)
        
    criterion = nn.ModuleList([])
    criterion.append(torch.nn.CrossEntropyLoss())
    criterion.append(KLLoss(opt.temperature))
    
    # Note that student and teacher feature shape is same
    if opt.method in ['ce', 'subnet_ce', 'kd']:
        criterion.append(None)
    elif opt.method == 'selfcon':
        criterion.append(ConLoss(temperature=opt.temperature))
    else:
        raise NotImplemented

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        
    return model, criterion, opt


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_s = AverageMeter()

    only_backbone = True if eval(opt.selfcon_pos) in [[False], [False,False], [False,False,False], [False,False,False,False]] else False
    
    end = time.time()
    for idx, inputs in enumerate(train_loader):
        images, labels = inputs
        
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.method not in ['ce', 'subnet_ce', 'kd']:
            feats, logits = model(images)
        else:
            logits = model(images)        
        
        loss = criterion[0](logits[-1], labels)
        
        for sub_logit in logits[0]:
            loss += opt.alpha * criterion[0](sub_logit, labels)
            loss += opt.beta * criterion[1](sub_logit, logits[-1])
        if criterion[2] is not None:
            for idx, feat_s in enumerate(feats[0]):
                # MLP head of backbone is always in random intialization
                features = torch.cat([feat_s.unsqueeze(1), feats[-1].unsqueeze(1)], dim=1)
                loss += opt.gamma * criterion[2](features, labels)
        
        # update metric
        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(logits[-1], labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        if not only_backbone:
            acc1_s, _ = accuracy(logits[0][0], labels, topk=(1, 5))
            top1_s.update(acc1_s[0], bsz)
        else:
            top1_s.update(torch.tensor(0.0).to(acc1[0].device), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.avg:.3f} {top1_s.avg:.3f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top1_s=top1_s))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_b = AverageMeter()
    top5_b = AverageMeter()
    top1_s = AverageMeter()
    top5_s = AverageMeter()
    
    only_backbone = True if eval(opt.selfcon_pos) in [[False], [False,False], [False,False,False], [False,False,False,False]] else False
    
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if opt.method not in ['ce', 'subnet_ce', 'kd']:
                _, logits = model(images)
            else:
                logits = model(images) 
                
            loss = criterion[0](logits[-1], labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits[-1], labels, topk=(1, 5))
            top1_b.update(acc1[0], bsz)
            top5_b.update(acc5[0], bsz)
            if only_backbone:
                top1_s.update(torch.tensor(0.0).to(acc1[0].device), bsz)
                top5_s.update(torch.tensor(0.0).to(acc5[0].device), bsz)
            else:
                # only for the first sub-network (actually we use 1 sub-network)
                acc1_s, acc5_s = accuracy(logits[0][0], labels, topk=(1, 5))
                top1_s.update(acc1_s[0], bsz)
                top5_s.update(acc5_s[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 ({top1_b.avg:.3f}) ({top1_s.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1_b=top1_b, top1_s=top1_s))

    print(' * Acc@1 {top1_b.avg:.3f} {top1_s.avg:.3f}'.format(top1_b=top1_b, top1_s=top1_s))
    return losses.avg, top1_b.avg, top5_b.avg, top1_s.avg, top5_s.avg


def main():
    opt = parse_option()
    
    # fix seed
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.deterministic = True
    
    # build model and criterion
    model, criterion, opt = set_model(opt)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        opt.start_epoch = 1

    # warm-up for large-batch training,
    if opt.batch_size >= 1024:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # training routine
    best_acc1 = 0
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluation
        loss, val_acc1, val_acc5, val_acc1_s, val_acc5_s = validate(val_loader, model, criterion, opt)

        if val_acc1.item() > best_acc1:
            best_acc1 = val_acc1
            best_acc5 = val_acc5
            best_acc1_s = val_acc1_s
            best_acc5_s = val_acc5_s
            best_model = model.state_dict()

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, epoch, save_file)
    
    # save the best model
    # Note that accuracy in results.json is different from the saved best model
    # because of multiprocessing distributed setting
    model.load_state_dict(best_model)
    save_file = os.path.join(
        opt.save_folder, 'best.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
        
    update_json_list(opt.save_folder, [best_acc1.item(), best_acc5.item(), best_acc1_s.item(), best_acc5_s.item(), train_acc.item()], path='./save/distill/results.json')


if __name__ == '__main__':
    main()
