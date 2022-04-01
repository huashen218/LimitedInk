#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils_dataloader import to_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model_Loss(nn.Module):

    def __init__(self, k, num_classes = 2, **model_kwargs):
        super(Model_Loss, self).__init__()
        self.k = k
        self.model_kwargs = model_kwargs
        self.num_classes = num_classes


    def forward(self, outputs_support, outputs_delete, targets, position_mask, masks):   # masks shape: [1]  targets shape: [batch_size]

        comprehensive_logits = outputs_delete[1]
        one_hot_targets = self.one_hot(targets)
        comprehensive_loss = self.comprehensive_norm(comprehensive_logits, one_hot_targets)

        support_loss = outputs_support[0]
        continuity_loss = self.continuity_norm(masks)
        sparsity_loss = self.sparsity_norm(masks, position_mask)

        total_loss = support_loss + \
                    self.model_kwargs['comprehensive_lambda'] * comprehensive_loss +\
                    self.model_kwargs['continuity_lambda'] * continuity_loss +\
                    self.model_kwargs['sparsity_lambda'] * sparsity_loss

        return total_loss



    def sparsity_norm(self, mask, position_mask): # mask shape: (batch_size, sent_len): (64, 133)
        reference = torch.zeros(mask.shape).to(device)
        for n in range(len(mask)):
            reference[n, -int(torch.sum(position_mask[n]).item() * self.k):] = 1
        mask_sorted = torch.sort(mask, dim=-1)[0]
        area_norm = torch.sum(torch.norm((mask_sorted - reference), p=1, dim=1))
        return area_norm


    def continuity_norm(self, mask):
        l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
        continuity_norm = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
        return continuity_norm


    def one_hot(self, targets):
        depth = self.num_classes
        if targets.is_cuda:
            return Variable(torch.zeros(targets.size(0), depth).cuda().scatter_(1, targets.long().view(-1, 1).data, 1))
        else:
            return Variable(torch.zeros(targets.size(0), depth).scatter_(1, targets.long().view(-1, 1).data, 1))


    def comprehensive_norm(self, output_delete, one_hot_targets):
        return self.cw_loss(output_delete, one_hot_targets, targeted=False, t_conf=1., nt_conf=5)


    def cw_loss(self, logits, one_hot_labels, targeted=True, t_conf=2, nt_conf=5):   # one_hot_labels shape: torch.Size([8, 2]);  logit shape: torch.Size([8])        
        this = torch.sum(logits * one_hot_labels, 1)
        other_best, _ = torch.max(logits*(1.-one_hot_labels) - 12111*one_hot_labels, 1)   # subtracting 12111 from selected labels to make sure that they dont end up a maximum
        t = F.relu(other_best - this + t_conf)
        nt = F.relu(this - other_best + nt_conf)
        if isinstance(targeted, (bool, int)):
            return torch.mean(t) if targeted else torch.mean(nt)





class Sparse_Loss(nn.Module):

    def __init__(self, k, num_classes = 2, **model_kwargs):
        super(Sparse_Loss, self).__init__()
        self.k = k
        self.model_kwargs = model_kwargs
        self.num_classes = num_classes


    def forward(self, outputs_support, outputs_delete, targets, masks):
        sparse_loss = self.sparse_norm(masks)
        continuity_loss = self.continuity_norm(masks)
        support_loss = outputs_support[0]
        total_loss = support_loss + self.model_kwargs['sparse_norm'] * (sparse_loss + continuity_loss)
        return total_loss

    def sparse_norm(self, mask):
        selection_cost = torch.mean( torch.sum(mask, dim=1) )
        return selection_cost

    def continuity_norm(self, mask):
        l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
        continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
        return continuity_cost




class Control_Loss(nn.Module):

    def __init__(self, k, num_classes = 2, **model_kwargs):
        super(Control_Loss, self).__init__()
        self.k = k
        self.model_kwargs = model_kwargs
        self.num_classes = num_classes


    def forward(self, outputs_support, outputs_delete, targets, masks):
        support_loss = outputs_support[0]
        control_loss = self.control_norm(masks)
        total_loss = support_loss + self.model_kwargs['control_norm'] * control_loss
        return total_loss


    def control_norm(self, mask):
        attention_mask = np.zeros(mask.size())
        for i in range(mask.size(0)):
            threshold = torch.sort(torch.abs(mask[i]), -1)[0][int(mask.size(-1) * (1-self.k))]
            attention_mask[i] = np.where(torch.abs(mask[i]).detach().cpu().numpy() > threshold.item(), torch.abs(mask[i]).detach().cpu().numpy(), 0)
        control_norm = torch.sum(to_tensor(attention_mask))
        return control_norm

