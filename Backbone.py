import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import math, time
from tabulate import tabulate
from torchvision import datasets, models, transforms
from torch.nn import Parameter
import os, time, glob
from datetime import datetime
from itertools import cycle
import numpy as np
import traceback
from utils import accuracy
from others_cosface import MarginCosineProduct

class OECNN_Backbone(nn.Module):
    def __init__(self, init_method, drop_rate=0):
        super().__init__()
        mods = nn.ModuleList()
        mods.append(BN_Conv_ReLU(3, 64))
        mods.append(BN_Conv_ReLU(64, 64))
        mods.append(nn.MaxPool2d(kernel_size=2))
        for _ in range(3):
            mods.append(Block(64, 64))
        mods.append(BN_Conv_ReLU(64, 128))
        mods.append(nn.MaxPool2d(kernel_size=2))
        for _ in range(4):
            mods.append(Block(128, 128))
        mods.append(BN_Conv_ReLU(128, 256))
        mods.append(nn.MaxPool2d(kernel_size=2))
        for _ in range(10):
            mods.append(Block(256, 256))
        mods.append(BN_Conv_ReLU(256, 512))
        mods.append(nn.MaxPool2d(kernel_size=2))
        for _ in range(3):
            mods.append(Block(512, 512))
        BN = nn.BatchNorm2d(512)
        flatten = nn.Flatten()
        dropout = nn.Dropout(drop_rate)
        fc = nn.Linear(512*7*6, 512)
        final_BN = nn.BatchNorm1d(512)
        self.seq = nn.Sequential(*mods, BN, flatten, dropout, fc, final_BN)
        # initialize paras
        if init_method is not None:
            for mod in self.seq.modules():
                if isinstance(mod, nn.Conv2d):
                    init_method['method'](mod.weight, **init_method['paras'])
                    if mod.bias is not None:
                        mod.bias.data.fill_(0)
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.constant_(mod.weight, 1)
                elif isinstance(mod, nn.Linear):
                    init_method['method'](mod.weight, **init_method['paras'])
                    if mod.bias is not None:
                        mod.bias.data.fill_(0)
        #time.sleep(1000)
    def __str__(self):
        return 'OECNN'

    def forward(self, x):
        return self.seq(x).renorm(2,0,1e-5).mul(1e5)
        
def BN_Conv_ReLU(n_in, n_out, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(nn.BatchNorm2d(n_in), nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), 
                         nn.ReLU(inplace=True))

class Block(nn.Module):
    '''
    Basic building block of OECNN (OECNN_Backbone), originally introduced in OECNN paper.
    '''
    def __init__(self, n_in, n_out, filter_sz=3, stride=1, block_sz=3):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.filter_sz = filter_sz
        self.stride = stride
        self.block_sz = block_sz
        mods = []
        mods.extend([nn.BatchNorm2d(n_in), nn.Conv2d(n_in, n_out, kernel_size=filter_sz, stride=self.stride, padding=1, bias=True), nn.ReLU(inplace=True)])
        for _ in range(block_sz-1):
            mods.extend([nn.BatchNorm2d(n_out), nn.Conv2d(n_out, n_out, kernel_size=filter_sz, stride=self.stride, padding=1, bias=True), nn.ReLU(inplace=True)])
        self.seq = nn.Sequential(*mods)

    def forward(self, x):
        return self.seq(x) + x


class ArcfaceMargin(nn.Module):
    def __init__(self, n_cls, embedding_size, s=64., m=0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(n_cls, embedding_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, xs, ys):
        logits = F.linear(F.normalize(xs), F.normalize(self.weight))
        if not self.training:
            return logits
        return logits.scatter(1, ys.view(-1, 1), (logits.gather(1, ys.view(-1, 1)).acos() + self.m).cos()).mul(self.s)

class CosMargin_v2(nn.Module):
    '''
    Loss defined in Cosface paper.
    '''
    def __init__(self, classnum, embedding_size=512,  s=64., m=0.35):
        super().__init__()
        self.m = m
        self.s = s
        self.fc = nn.Linear(embedding_size, classnum, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, xs, labels):
        coses = self.fc(xs.renorm(2, 0, 1e-5).mul(1e5))
        if not self.training:
            return coses
        return coses.scatter_add(1, labels.view(-1,1), coses.new_full(labels.view(-1,1).size(), -self.m)).mul(self.s)

class CosMargin(nn.Module):
    '''
    Loss defined in Cosface paper.
    '''
    def __init__(self, classnum, embedding_size=512,  s=64., m=0.35):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, xs, labels):
        coses = F.linear(F.normalize(xs), F.normalize(self.weight))
        if not self.training:
            return coses
        return coses.scatter_add(1, labels.view(-1,1), coses.new_full(labels.view(-1,1).size(), -self.m)).mul(self.s)


class RFM(nn.Module):
    '''
    Residual Factorization Module in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(n_in, n_in)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(n_in, n_in)
                                , nn.ReLU(inplace=True))
    
    def forward(self, xs):
        return self.seq(xs)


class DAL_regularizer(nn.Module):
    '''
    Decorrelated Adversarial Learning module in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.w_age = nn.Linear(n_in, 1, bias=False)
        self.w_id = nn.Linear(n_in, 1, bias=False)
    
    def forward(self, features_age, features_id):
        vs_age = self.w_age(features_age)
        vs_id = self.w_id(features_id)
        rho = ((vs_age - vs_age.mean(dim=0)) * (vs_id - vs_id.mean(dim=0))).mean(dim=0).pow(2) \
                / ( (vs_age.var(dim=0) + 1e-6) * (vs_id.var(dim=0) + 1e-6))
        return rho




class DAL_model(nn.Module):
    '''
    The final ensemble model for training.
    '''
    def __init__(self, head, n_cls, embedding_size=512, init_method={'method': nn.init.kaiming_normal_, 'paras':{}}):
        super().__init__()
        self.backbone = OECNN_Backbone(init_method)
        if head.lower() in 'cosface':
            self.margin_fc = CosMargin(n_cls, embedding_size=512, s=64.,m=0.35)  # 32 0.1 worked
        elif head.lower() in 'arcface':
            self.margin_fc = ArcfaceMargin(n_cls, embedding_size)
        self.DAL = DAL_regularizer(embedding_size)
        self.RFM = RFM(embedding_size)
        self.age_classifier = nn.Sequential(nn.Linear(embedding_size, embedding_size) \
                                            , nn.ReLU(inplace=True)
                                            , nn.Linear(embedding_size, embedding_size)
                                            , nn.ReLU(inplace=True)
                                            , nn.Linear(embedding_size, 8))
        self.id_cr = nn.CrossEntropyLoss()
        self.age_cr = nn.CrossEntropyLoss()

    def forward(self, xs, ys=None, agegrps=None, emb=False):
        # 512-D embedding
        embs = self.backbone(xs)
        embs_age = self.RFM(embs)
        embs_id = (embs - embs_age)
        if emb:
            return F.normalize(embs_id)
        # ID identifier
        logits = self.margin_fc(embs_id, ys)
        #print(f'logits:\n{logits.size()}\n{logits}')
        id_acc = accuracy(torch.max(logits, dim=1)[1], ys)
        #print(f'id_acc:\n{id_acc.size()}\n{id_acc}')
        idLoss = self.id_cr(logits, ys)
        #print(f'idLoss:\n{idLoss.size()}\n{idLoss}')
        # age classifier
        age_logits = self.age_classifier(embs_age)
        #print(f'age_logits:\n{age_logits.size()}\n{age_logits}')
        age_acc = accuracy(torch.max(age_logits, dim=1)[1], agegrps)
        #print(f'age_acc:\n{age_acc.size()}\n{age_acc}')
        ageLoss = self.age_cr(age_logits, agegrps)
        #print(f'ageLoss:\n{ageLoss.size()}\n{ageLoss}')
        # DAL
        cano_cor = self.DAL(embs_age, embs_id)
        return idLoss, id_acc, ageLoss, age_acc, cano_cor