from Backbone import DAL_model
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch import autograd
import os
from custom_datasets import ImageFolderWithAgeGroup
from meta import age_cutoffs
from utils import Recorder
from itertools import chain

class Trainer():
    def __init__(self, model, dataset, ctx=-1, batch_size=128, optimizer='sgd', lambdas=[0.1, 0.1], print_freq=32):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.optbb = optim.SGD(chain(self.model.age_classifier.parameters(), 
                                    self.model.RFM.parameters(), 
                                    self.model.margin_fc.parameters(), 
                                    self.model.backbone.parameters()), lr=0.01, momentum=0.9)
        self.optDAL = optim.SGD(self.model.DAL.parameters(), lr=0.01, momentum=0.9)
        self.lambdas = lambdas
        self.print_freq = print_freq
        self.id_recorder = Recorder()
        self.age_recorder = Recorder()
        self.trainingDAL = False
        if ctx < 0:
            self.ctx = torch.device('cpu')
        else:
            self.ctx = torch.device(f'cuda:{ctx}')
    
    def train(self, epochs, start_epoch, save_path=None):
        self.train_ds = ImageFolderWithAgeGroup(self.dataset['pat'], self.dataset['pos'], \
                        age_cutoffs, self.dataset['train_root'], transform=transforms.Compose(\
                            [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), \
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
        self.train_ld = DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)
        if self.dataset['val_root'] is not None:
            self.val_ds = ImageFolderWithAgeGroup(self.dataset['pat'], self.dataset['pos'], age_cutoffs, self.dataset['val_root'], \
                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
            self.val_ld = DataLoader(self.val_ds, shuffle=True, batch_size=self.batch_size)
        self.model = self.model.to(self.ctx)
        for epoch in range(epochs):
            print(f'---- epoch {epoch} ----')
            self.update()
            if self.dataset['val_root'] is not None:
                acc = self.validate()
            else:
                acc = -1.
            if save_path is not None:
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{start_epoch+epoch}_{acc:.4f}.state'))

    def update(self):
        print('    -- Training --')
        self.model.train()
        self.id_recorder.reset()
        self.age_recorder.reset()
        for i, (xs, ys, agegrps) in enumerate(self.train_ld):
            if i % 70 == 0: # canonical maximization procesure
                self.set_train_mode(False)
            elif i % 70 == 20: #  RFM optimize procesure
                self.set_train_mode(True)
            xs, ys, agegrps = xs.to(self.ctx), ys.to(self.ctx), agegrps.to(self.ctx)
            idLoss, id_acc, ageLoss, age_acc, cc = self.model(xs, ys, agegrps)
            #print(f'        ---\n{idLoss}\n{id_acc}\n{ageLoss}\n{age_acc}\n{cc}')
            total_loss =  idLoss + ageLoss*self.lambdas[0] + cc*self.lambdas[1]
            self.id_recorder.gulp(len(agegrps), idLoss.item(), id_acc.item())
            self.age_recorder.gulp(len(agegrps), ageLoss.item(), age_acc.item())
            if i % self.print_freq == 0:
                print(f'        iter: {i} {i%70} total loss: {total_loss.item():.4f} ({idLoss.item():.4f}, {id_acc.item():.4f}, {ageLoss.item():.4f}, {age_acc.item():.4f}, {cc.item():.8f})')
            if self.trainingDAL:
                self.optDAL.zero_grad()
                total_loss.backward()
                Trainer.flip_grads(self.model.DAL)
                self.optDAL.step()
            else:
                self.optbb.zero_grad()
                total_loss.backward()
                self.optbb.step()
        # show average training meta after epoch
        print(f'        {self.id_recorder.excrete().result_as_string()}')
        print(f'        {self.age_recorder.excrete().result_as_string()}')

    def validate(self):
        print('    -- Validating --')
        self.model.eval()
        self.id_recorder.reset()
        self.age_recorder.reset()
        for i, (xs, ys, agegrps) in enumerate(self.val_ld):
            xs, ys, agegrps = xs.to(self.ctx), ys.to(self.ctx), agegrps.to(self.ctx)
            with torch.no_grad():
                idLoss, id_acc, ageLoss, age_acc, cc = self.model(xs, ys, agegrps)
                total_loss =  idLoss + ageLoss*self.lambdas[0] + cc*self.lambdas[1]
                self.id_recorder.gulp(len(agegrps), idLoss.item(), id_acc.item())
                self.age_recorder.gulp(len(agegrps), ageLoss.item(), age_acc.item())
        # show average validation meta after epoch
        print(f'        {self.id_recorder.excrete().result_as_string()}')
        print(f'        {self.age_recorder.excrete().result_as_string()}')
        return self.id_recorder.acc


    def set_train_mode(self, state):
        self.trainingDAL = not state
        Trainer.set_grads(self.model.RFM, state)
        Trainer.set_grads(self.model.backbone, state)
        Trainer.set_grads(self.model.margin_fc, state)
        Trainer.set_grads(self.model.age_classifier, state)
        Trainer.set_grads(self.model.DAL, not state)


    @staticmethod
    def set_grads(mod, state):
        for para in mod.parameters():
            para.requires_grad = state

    @staticmethod
    def flip_grads(mod):
        for para in mod.parameters():
            if para.requires_grad:
                para.grad = - para.grad
