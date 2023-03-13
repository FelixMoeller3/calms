from typing import Callable, List
import torch.nn as nn
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import Dataset,DataLoader,Subset
import random
from models import testConv,ResNet,BasicBlock
from collections import Counter
import os
import pickle

class BaseProcess:

    def __init__(self,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy, state_dir:str=None):
        self.al_strat = activeLearningStrategy
        self.cl_strat = continualLearningStrategy
        self.state_dir = state_dir


    def _train_cycle(self,train_set: Dataset,training_examples: List[int],loaders_dict: dict[str,DataLoader],batch_size:int,num_epochs:int,score_list:List[float]) -> None:
        '''
            Trains the substitute model with the data queried in the current cycle.
            :param train_set: The full training dataset.
            :param training_examples: A list of indexes in the training set that the model should be trained on.
            :param loaders_dict: A dictionary containing the Dataloader for training data and one for the validation data. The can be accessed
            by the keywords 'train' and 'val'.
        '''
        training_set = Subset(train_set,training_examples)
        loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
        self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list)

    def _save_state(self,state_dict: dict):
        os.makedirs(self.state_dir,exist_ok=True)
        with open(os.path.join(self.state_dir, 'latest_state.pkl'), 'wb') as f :
                pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)
        torch.save(self.cl_strat.model.state_dict(),os.path.join(self.state_dir,"model.pth"))
