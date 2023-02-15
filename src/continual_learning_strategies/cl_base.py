from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ContinualLearningStrategy(ABC):
    '''
        This is the base class for all continual learning strategies
    '''
    def __init__(self, model:nn.Module,optim: torch.optim.Optimizer,crit: nn.CrossEntropyLoss):
        self.model = model
        self.optim = optim
        self.crit = crit

    @abstractmethod
    def train(self, dataloader: DataLoader):
        pass

    def save(self, filename: str):
        torch.save(self.model, filename)

    def load(self, filename: str):
        self.model = torch.load(filename)