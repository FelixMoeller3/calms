import torch
from .cl_base import ContinualLearningStrategy
from torch import nn
from torch.utils.data import Dataset
from typing import List, Optional
import random
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler

class IMM(ContinualLearningStrategy):
    '''
    Implementation of Incremental Moment Matching (IMM) according.
    '''

    def __init__(self, model: nn.Module,optim: torch.optim.Optimizer, scheduler: lr_scheduler._LRScheduler, crit: nn.CrossEntropyLoss,ALPHAS:List[float]=None,WEIGHT:float=1.0,MEAN:bool=True,USE_GPU:bool=False,**kwargs):
        '''
            :param alphas: List of weights for models of previous tasks,
             i.e. how strongly previous tasks should be weighted. The sum of all entries in this list must be 1.
             If no list is provided the all previous tasks will be weighted equally.
            :param mean: Whether to use mean-IMM or mode-IMM
        '''
        super(IMM,self).__init__(model,optim,scheduler,crit,USE_GPU)
        assert not ALPHAS or abs(sum(ALPHAS)-1.0) < 1e-8
        if ALPHAS:
            self.alphas = [torch.tensor(val).cuda() if self.use_gpu else torch.tensor(val) for val in ALPHAS]
        else:
            self.alphas = ALPHAS
        self.weight = torch.tensor(WEIGHT).cuda() if self.use_gpu else torch.tensor(WEIGHT)
        self.mean = MEAN
        self.num_tasks = 0
        self.prev_param_list = []
        self._save_model_params()
        self.prev_fishers = []

    def _save_model_params(self) -> None:
        prev_params = {}
        for name, param in self.model.named_parameters():
            prev_params[name] = param.detach().clone()
        self.prev_param_list.append(prev_params)
        # delete the first model in the list if its alpha value would be 0
        # because the list of previous models is longer than the list of alpha values
        if self.alphas and len(self.prev_param_list) > len(self.alphas):
            self.prev_param_list.pop(0)


    def _before_train(self) -> None:
        self._set_model_params()

    def _after_train(self,train_set: Dataset) -> None:
        self._save_model_params()
        self._merge_models(train_set)

    def _set_model_params(self) -> None:
        for name,param in self.model.named_parameters():
            param.data = self.prev_param_list[-1][name]
    
    def _compute_consolidation_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name,param in self.model.named_parameters():
            diff = param - self.prev_param_list[-1][name]
            diff.pow_(2)
            loss += diff.sum()
        return loss * self.weight

    def _merge_models(self, train_dataset:Optional[Dataset], sample_size:Optional[float]=0.05) -> None:
        '''
            TODO: Add method description
        '''
        new_model_weights = {}
        for name,param in self.model.named_parameters():
            new_model_weights[name] = torch.zeros_like(param)
        if self.alphas:
            alphas = self.alphas
        else:
            val = 1/len(self.prev_param_list)
            alphas = [torch.tensor(val).cuda() if self.use_gpu else torch.tensor(val)] * len(self.prev_param_list)
        if self.mean:
            for i,weights in enumerate(self.prev_param_list):
                for name,param in self.model.named_parameters():
                    new_model_weights[name] += alphas[i] * weights[name]

        else:
            '''
                TODO: The numbers seem odd for mode-IMM. Check if calculation works 
            '''
            self._calc_fisher(train_dataset,sample_size)
            sigma = self._calc_sigma()
            for i,(weights,fisher) in enumerate(zip(self.prev_param_list,self.prev_fishers)):
                for name,param in self.model.named_parameters():
                    new_model_weights[name] += alphas[i] * weights[name] * fisher[name]
            for param_name in new_model_weights:
                new_model_weights[param_name] *= sigma[param_name]
                
        for name,param in self.model.named_parameters():
            param.data = new_model_weights[name]

        
    def _calc_fisher(self,train_dataset: Dataset=None, sample_size:float=0.05) -> None:
        '''
            TODO: Add method description
        '''
        num_samples = (len(train_dataset) * sample_size) if train_dataset else 0
        fisher = {}
        for name,param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        for _ in range(int(num_samples)):
            cur_index = random.randint(0,len(train_dataset)-1)
            elem, label = train_dataset[cur_index]
            if not isinstance(elem,torch.Tensor):
                elem = torch.tensor(elem)
            if self.use_gpu:
                elem = elem.cuda()
            if isinstance(label,torch.Tensor):
                class_label = torch.argmax(label).item()
            else:
                class_label = label
            self.optim.zero_grad()
            output = self.model(elem)
            sm = F.log_softmax(output,dim=1)
            label_tensor = torch.tensor([class_label],dtype=torch.long).cuda() if self.use_gpu else torch.tensor([class_label],dtype=torch.long)
            loss = F.nll_loss(sm,label_tensor)
            loss.backward()

            for name, param in self.model.named_parameters():
                fisher[name] += param.grad.data ** 2 / num_samples
                # add small epsilon to fisher values to make sure the are greater than zero.
                # This is necessary because the values will be inverted later
                fisher[name] += 1e-8

        self.prev_fishers.append(fisher)
        # delete the first fisher matrix in the list if its alpha value would be 0
        # because the list of previous fisher matrices is longer than the list of alpha values
        if self.alphas and len(self.prev_fishers) > len(self.alphas):
            self.prev_fishers.pop(0)
        

    def _calc_sigma(self) -> dict[str,torch.Tensor]:
        '''
            TODO: Add method description
        '''
        sigma = {}
        for name,param in self.model.named_parameters():
            sigma[name] = torch.zeros_like(param)
        alphas =  [1/len(self.prev_param_list)] * len(self.prev_param_list) if not self.alphas else self.alphas
        for name,param in self.model.named_parameters():
            for i,weights in enumerate(self.prev_fishers):
                sigma[name] += alphas[i-1] * weights[name]
                sigma[name] = 1/sigma[name]

        return sigma
