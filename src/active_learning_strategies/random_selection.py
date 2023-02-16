from .strategy import Strategy
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class RandomSelection(Strategy):
    '''
        Random selection strategy
    '''
    def __init__(self, model: nn.Module, data_unlabeled: Dataset, NO_CLASSES: int,BATCH:int,BUDGET:int,INIT_BUDGET:int, device=None,**kwargs):
        super(RandomSelection, self).__init__(model, data_unlabeled, NO_CLASSES,BATCH,BUDGET,INIT_BUDGET,device)

    def query(self) -> np.ndarray:
        random.seed(self.init_set_seed)
        arg = list(range(len(self.data_unlabeled)))
        random.shuffle(arg)
        arg = np.array(arg)
        return arg[:self.BUDGET]
    