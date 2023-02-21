from typing import List
from .cl_base import ContinualLearningStrategy
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Naive(ContinualLearningStrategy):

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer,criterion: torch.nn.CrossEntropyLoss,USE_GPU:bool=False,**kwargs):
        super(Naive,self).__init__(model,optimizer,criterion,USE_GPU)
