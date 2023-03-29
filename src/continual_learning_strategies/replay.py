from .cl_base import ContinualLearningStrategy
from torch import nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
import random
from torch.utils.data import Dataset,DataLoader,Subset
from typing import List

class Replay(ContinualLearningStrategy):

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer,scheduler: lr_scheduler._LRScheduler,criterion: torch.nn.CrossEntropyLoss,USE_GPU:bool=False,PATTERNS_PER_EXPERIENCE: int=300,**kwargs):
        super(Replay,self).__init__(model,optimizer,scheduler,criterion,USE_GPU)
        self.patterns_per_experience = PATTERNS_PER_EXPERIENCE
        self.indices = []

    # def _get_memory_sample(self) -> List[int]:
    #     if len(self.indices) < self.mem_size:
    #         return self.indices
    #     else:
    #         return random.sample(self.indices,self.mem_size)

    def train(self,dataloaders: dict[str,DataLoader],num_epochs:int,val_step:int,result_list:List[float]=[],early_stopping:int=-1) -> None:
        if self.isActive or not isinstance(dataloaders["train"].dataset,Subset):
            super(Replay,self).train(dataloaders,num_epochs,val_step,result_list,early_stopping)
        else:
            subset = dataloaders["train"].dataset
            indices = subset.indices + self.indices
            dataloaders["train"].dataset = Subset(subset.dataset,indices)
            super(Replay,self).train(dataloaders,num_epochs,val_step,result_list,early_stopping)



    def _after_train(self,train_set: Dataset=None) -> None:
        if not isinstance(train_set,Subset):
            return
        self.indices += random.sample(train_set.indices,self.patterns_per_experience)