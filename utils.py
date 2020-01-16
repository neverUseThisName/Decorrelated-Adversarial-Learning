import re, sys, os
from os.path import basename
import numpy as np
import torch

def path2age(path, pat, pos):
    return int(re.split(pat, basename(path))[pos])

def accuracy(preds, labels):
    return (preds.squeeze()==labels.squeeze()).float().mean()

def erase_print(content):
    sys.stdout.write('\033[2K\033[1G')
    sys.stdout.write(content)
    sys.stdout.flush()

def mkdir_p(path):
    try:
        os.makedirs(os.path.abspath(path))
    except OSError as exc: 
        if exc.errno == os.errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def get_ctx(n):
    return torch.device(f'cuda:{n}') if n >=0 else torch.device('cpu')

class Recorder():
    def __init__(self):
        self.N = 0
        self.loss = 0
        self.n_crct = 0

    def reset(self):
        self.N = 0
        self.loss = 0
        self.n_crct = 0

    def gulp(self, n, loss, acc):
        self.N += n
        self.loss += n*loss
        self.n_crct += int(n*acc)

    def excrete(self):
        self.loss = self.loss / self.N
        self.acc = self.n_crct / self.N
        return self
    
    def result_as_string(self):
        return f'{self.N}, {self.loss:.4f}, {self.acc:.4f}'