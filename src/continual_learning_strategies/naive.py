from typing import List
from .cl_base import ContinualLearningStrategy
from torch import nn
import torch
import torch.optim.lr_scheduler as lr_scheduler

class Naive(ContinualLearningStrategy):

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer,scheduler: lr_scheduler._LRScheduler,criterion: torch.nn.CrossEntropyLoss,USE_GPU:bool=False,**kwargs):
        super(Naive,self).__init__(model,optimizer,scheduler,criterion,USE_GPU)
