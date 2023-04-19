from .cl_base import ContinualLearningStrategy
from torch import nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
import random
from torch.utils.data import Dataset,DataLoader,Subset
from typing import List
from active_learning_strategies import CoreSet,RandomSelection

class Replay(ContinualLearningStrategy):

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer,scheduler: lr_scheduler._LRScheduler,criterion: torch.nn.CrossEntropyLoss,dataset:Dataset,
                 USE_GPU:bool=False,BUFFER_SIZE:int=2000,state_dict:dict=None,**kwargs):
        super(Replay,self).__init__(model,optimizer,scheduler,criterion,USE_GPU)
        if state_dict is not None:
            self.set_state(state_dict)
            return
        self.buffer_size = BUFFER_SIZE
        self.indices = []
        #self.buffer_selection_strategy = RandomSelection(model,dataset,0,128,self.buffer_size,0,1,USE_GPU)
        self.buffer_selection_strategy = CoreSet(model,dataset,0,128,self.buffer_size,0,1,USE_GPU)

    # def _get_memory_sample(self) -> List[int]:
    #     if len(self.indices) < self.mem_size:
    #         return self.indices
    #     else:
    #         return random.sample(self.indices,self.mem_size)

    def train(self,dataloaders: dict[str,DataLoader],num_epochs:int,val_step:int,result_list:List[float]=[],early_stopping:int=-1) -> None:
        if not self.isActive or not isinstance(dataloaders["train"].dataset,Subset):
            super(Replay,self).train(dataloaders,num_epochs,val_step,result_list,early_stopping)
        else:
            subset = dataloaders["train"].dataset
            self.cur_subset_len = len(subset)
            indices = subset.indices + self.indices
            dataloaders["train"] = DataLoader(Subset(subset.dataset,indices),dataloaders["train"].batch_size,shuffle=True)
            super(Replay,self).train(dataloaders,num_epochs,val_step,result_list,early_stopping)



    def _after_train(self,train_set: Dataset=None) -> None:
        if not isinstance(train_set,Subset):
            return
        self.indices += train_set.indices[:self.cur_subset_len]
        if len(self.indices) <= self.buffer_size:
            return
        self.buffer_selection_strategy.feed_current_state(0,self.indices,[])
        samples = self.buffer_selection_strategy.query()
        self.indices = [self.indices[i] for i in samples]

    def get_state(self) -> dict:
        return {
            'buffer_size': self.buffer_size,
            'indices': self.indices,
        }
    
    def set_state(self,state:dict) -> None:
        self.buffer_size = state['buffer_size']
        self.indices = state['indices']
        self.buffer_selection_strategy.feed_current_state(0,self.indices,[])