


import sys
import torch
from Backbone import DAL_model
from train import Trainer
from meta import mixture, FGNET, vgg_toy, age_cutoffs
from utils import path2age
# pytohn3 main.py ctx_id epochs path2wgts      # ctx_id: -1 for cpu, 0 for gpu 0

def main():
    dataset = mixture
    model = DAL_model('cosface', dataset['n_cls'])
    if len(sys.argv) >= 4:
        model.load_state_dict(torch.load(sys.argv[3]))
        print(f'Loaded weights: {sys.argv[3]}')
        start_epoch = path2age(sys.argv[3], '_|\.', 0) + 1
    else:
        start_epoch = 0
    trainer = Trainer(model, dataset, int(sys.argv[1]), print_freq=1)
    save = '/data/fuzhuolin/DAL/state_dicts/1'
    trainer.train(int(sys.argv[2]), start_epoch, save)

if __name__ == '__main__':
    main()