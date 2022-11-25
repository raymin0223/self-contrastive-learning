"""
refer to 
1) Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
2) SimCLR: https://arxiv.org/pdf/2002.05709.pdf
"""
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7


class ConLoss(nn.Module):
    """Self-Contrastive Learning: https://arxiv.org/abs/2106.15499."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, supcon_s=False, selfcon_s_FG=False, selfcon_m_FG=False):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            supcon_s: boolean for using single-viewed batch.
            selfcon_s_FG: exclude contrastive loss when the anchor is from F (backbone) and the pairs are from G (sub-network).
            selfcon_m_FG: exclude contrastive loss when the anchor is from F (backbone) and the pairs are from G (sub-network).
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] if not selfcon_m_FG else int(features.shape[0]/2)
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)    
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        if not selfcon_s_FG and not selfcon_m_FG:
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            if self.contrast_mode == 'one':
                anchor_feature = features[:, 0]
                anchor_count = 1
            elif self.contrast_mode == 'all':
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        elif selfcon_s_FG:
            contrast_count = features.shape[1]
            anchor_count = features.shape[1]-1
            
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        elif selfcon_m_FG:
            contrast_count = int(features.shape[1] * 2)
            anchor_count = (features.shape[1]-1)*2
            
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
            
        mask = mask * logits_mask
        if supcon_s:
            idx = mask.sum(1) != 0
            mask = mask[idx, :]
            logits_mask = logits_mask[idx, :]
            logits = logits[idx, :]
            batch_size = idx.sum()
            
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
            
        return loss


class KLLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=3.0):
        super(KLLoss, self).__init__()
        self.T = T

    def forward(self, logit_s, logit_t):
        p_s = F.log_softmax(logit_s/self.T, dim=1)
        p_t = F.softmax(logit_t.clone().detach()/self.T, dim=1)
        loss = -pow(self.T, 2)*(p_s * p_t).sum(dim=1).mean()
        
        return loss