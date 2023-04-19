from typing import List, Callable
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import Dataset,DataLoader,Subset
import os
import pickle
import random
import time
import submodlib
import numpy as np
from functools import reduce

INIT_MODES = ["facility_location", "random"]

class BaseProcess:

    def __init__(self,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy, train_set: Dataset,
                 val_set: Dataset,batch_size:int,num_cycles:int,num_epochs:int,continual:int,optimizerBuilder: Callable, optimizerConfig: dict,
                 init_mode:str='random',use_gpu:bool=False,cold_start:bool=False,state_dir:str=None):
        self.al_strat = activeLearningStrategy
        self.cl_strat = continualLearningStrategy
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.num_cycles = num_cycles
        self.num_epochs = num_epochs
        self.continualStart = continual
        self.optimizer_builder = optimizerBuilder
        self.optimizer_config = optimizerConfig
        self.init_mode = init_mode
        self.use_gpu = use_gpu
        self.state_dir = state_dir
        self.cold_start = cold_start and continual > num_cycles

    def _before_first_cycle(self,loaders_dict:dict[str,DataLoader],val_accuracies: List[int]) -> tuple[List[int],List[int]]:
        if self.continualStart > -1:
            self.cl_strat.deactivate()
        unlabeled_set = [i for i in range(len(self.train_set))]
        if self.init_mode == "facility_location":
            labeled_set = self._init_facility_location()
        elif self.init_mode =="random":
            labeled_set = self._init_random()
        else:
            raise ValueError(f"Got unknown mode {self.init_mode} as initialization mode. Mode must be one of {','.join(INIT_MODES)}")
        self._add_targets(labeled_set)
        self._train_cycle(self.train_set,labeled_set,loaders_dict,self.batch_size,self.num_epochs,val_accuracies)
        unlabeled_set = [i for i in unlabeled_set if i not in labeled_set]
        return labeled_set,unlabeled_set

    def _add_targets(self,labeled_set: List[int]) -> None:
        pass

    def _query_cycle(self, cycle_number: int, labeled_set: List[int], unlabeled_set: List[int], loaders_dict: dict[str,DataLoader],val_accuracies: List[float]) -> tuple[List[int],List[int]]:
        if cycle_number == self.continualStart:
            print("Switching from pure active learning to continual active learning")
            self.cl_strat.activate()
        self.al_strat.feed_current_state(cycle_number,unlabeled_set,labeled_set)
        training_examples = self.al_strat.query()
        training_examples_absolute_indices = [unlabeled_set[elem] for elem in training_examples]
        self._add_targets(training_examples_absolute_indices)
        labeled_set += training_examples_absolute_indices
        unlabeled_set = [i for i in unlabeled_set if i not in training_examples_absolute_indices]
        cur_train_set = training_examples_absolute_indices if self.continualStart <= cycle_number else labeled_set
        self._train_cycle(self.train_set,cur_train_set,loaders_dict,self.batch_size,self.num_epochs,val_accuracies)
        return labeled_set,unlabeled_set

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
        if self.cold_start:
            self.cl_strat.model.weight_reset()
        self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list)
        optim,scheduler = self.optimizer_builder(self.optimizer_config,self.cl_strat.model)
        self.cl_strat.optim = optim
        self.cl_strat.scheduler = scheduler

    def _save_state(self,state_dict: dict):
        os.makedirs(self.state_dir,exist_ok=True)
        with open(os.path.join(self.state_dir, 'latest_state.pkl'), 'wb') as f :
                pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)
        torch.save(self.cl_strat.model.state_dict(),os.path.join(self.state_dir,"model.pth"))

    def _init_facility_location(self) -> List[int]:
        print("Running facility location")
        num_features = reduce(lambda x,y:x*y,self.train_set[0][0].shape,1)
        num_samples = len(self.train_set)
        #num_samples = 40000
        data = np.zeros((num_samples,num_features))
        for i in range(num_samples):
            data[i] = self.train_set[i][0].numpy().flatten()
        print("Finished initializing. Setting up fl space...")
        fl = submodlib.FacilityLocationFunction(n=num_samples,mode="dense",data=data,metric='cosine')
        print("Trying to get fl set...")
        output_set = fl.maximize(self.al_strat.INIT_BUDGET)
        return [i for (i,_) in output_set]

    def _init_random(self) -> List[int]:
        labeled_set = [i for i in range(len(self.train_set))]
        random.shuffle(labeled_set)
        labeled_set = labeled_set[:self.al_strat.INIT_BUDGET]
        return labeled_set
