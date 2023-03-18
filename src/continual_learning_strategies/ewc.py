from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .cl_base import ContinualLearningStrategy
import torch.optim.lr_scheduler as lr_scheduler
import random
from torch.nn import functional as F


class ElasticWeightConsolidation(ContinualLearningStrategy):
    '''
        Implementation of Elastic Weight Consolidation (EWC) as proposed in the following paper:
        https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114
        The code is heavily based on the following implementations:
        https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks and
        https://github.com/thuyngch/Overcoming-Catastrophic-Forgetting
    '''

    def __init__(self,model:nn.Module,optim: torch.optim.Optimizer,scheduler: lr_scheduler._LRScheduler,crit: nn.CrossEntropyLoss,WEIGHT:float=1.0,SCHEDULE:bool=False,USE_GPU:bool=False,**kwargs):
        super(ElasticWeightConsolidation,self).__init__(model,optim,scheduler,crit,USE_GPU)
        self.weight = torch.tensor(WEIGHT).cuda() if self.use_gpu else torch.tensor(WEIGHT)
        self.prev_params = {}
        self._save_model_params()
        self.fisher = {}
        self._update_fisher_params()
        self.schedule_weight = SCHEDULE 
        self.n_tasks = 0

    def _save_model_params(self) -> None:
        '''
            Saves the current parameters of the model before training on a new task because the old 
            model parameters are needed to compute the loss function
        '''
        for name,param in self.model.named_parameters():
            self.prev_params[name] = param.detach().clone()

    def _after_train(self,train_set: Dataset) -> None:
        self._save_model_params()
        self._update_fisher_params(train_set,0.05)

    def _update_fisher_params(self, train_dataset: Dataset=None, sample_size:float=0.05):
        '''
            TODO: Add method description
        '''
        # TODO: Fix so we can work with class label when having softmax label
        num_samples = (len(train_dataset) * sample_size) if train_dataset else 0
        self.fisher = {}
        for name,param in self.model.named_parameters():
            self.fisher[name] = torch.zeros_like(param)
        for _ in range(int(num_samples)):
            cur_index = random.randint(0,len(train_dataset)-1)
            elem, label = train_dataset[cur_index]
            # Labels can be an array when they are the softmax output of the target model
            if isinstance(label,torch.Tensor):
                class_label = torch.argmax(label).item()
            else:
                class_label = label
            self.optim.zero_grad()
            input = torch.unsqueeze(elem,0)
            if self.use_gpu:
                input = input.cuda()
            output = self.model(input)
            sm = F.log_softmax(output,dim=1)
            label_tensor = torch.tensor([class_label],dtype=torch.long).cuda() if self.use_gpu else torch.tensor([class_label],dtype=torch.long)
            loss = F.nll_loss(sm,label_tensor)
            loss.backward()

            for name, param in self.model.named_parameters():
                self.fisher[name] += param.grad.data ** 2 / num_samples


    def _compute_regularization_loss(self):
        '''
            TODO: Add method description
        '''
        loss = torch.tensor(0.0).cuda() if self.use_gpu else torch.tensor(0.0)
        for name,param in self.model.named_parameters():
            diff:torch.Tensor = param - self.prev_params[name]
            diff.pow_(2)
            diff.mul_(self.fisher[name])
            loss += diff.sum()
        return loss * (self.weight / 2)
       
    def _update_weight(self) -> None:
        if not self.schedule_weight:
            return
        self.n_tasks += 1
        if self.n_tasks % 5 == 0:
            self.n_tasks = 0
            prev_weight = self.weight.item()
            self.weight = torch.tensor(2*prev_weight).cuda() if self.use_gpu else torch.tensor(2*prev_weight)
