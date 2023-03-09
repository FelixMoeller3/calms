from typing import Callable, List
import torch.nn as nn
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import Dataset,DataLoader,Subset
import random
from models import testConv,ResNet,BasicBlock
from collections import Counter

class BaseProcess:

    def __init__(self,targetModel:nn.Module,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy):
        pass