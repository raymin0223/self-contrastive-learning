from __future__ import print_function

import os
import json
import pickle
import math
import numpy as np
import torch
import torch.optim as optim

__all__ = ['AverageMeter', 'TwoCropTransform', 'adjust_learning_rate', 'warmup_learning_rate', 'accuracy', 'class_accuracy', 'set_optimizer', 'save_model', 'update_json', 'update_json_list']


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].reshape(-1, k).float().sum(1).sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def class_accuracy(output, target, cls, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        
        output = output[target == cls]
        target = target[target == cls]

        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].reshape(-1, k).float().sum(1).sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, batch_size


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model, optimizer='sgd'):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=opt.learning_rate)                               
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
    
    
def update_json(exp_name, acc={}, path='./save/results.json'):
    for k, v in acc.items():
        acc[k] = [round(a, 2) for a in v]
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)

    with open(path, 'r', encoding="UTF-8") as f:
        result_dict = json.load(f)
        result_dict[exp_name] = acc
    
    with open(path, 'w') as f:
        json.dump(result_dict, f)
        
    print('best accuracy: {}'.format(acc))
    print('results updated to %s' % path)
    

def update_json_list(exp_name, acc=[0., 0.], path='./save/results.json'):
    acc = [round(a, 2) for a in acc]
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)

    with open(path, 'r', encoding="UTF-8") as f:
        result_dict = json.load(f)
        result_dict[exp_name] = acc
    
    with open(path, 'w') as f:
        json.dump(result_dict, f)
        
    print('best accuracy: {}'.format(acc))
    print('results updated to %s' % path)