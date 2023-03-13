from .process_base import BaseProcess
import torch.nn as nn
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies import Naive
from torch.utils.data import Dataset,DataLoader
from typing import Callable,List
import random
from collections import Counter

class AlProcess(BaseProcess):
    def __init__(self,activeLearningStrategy: Strategy,trainingConfig: Naive,state_dir:str=None):
        '''
            :param targetModel: the target model in the model stealing process (i.e. the one that will be stolen). Needs to be pretrained!!
            :param activeLearningStrategy: the active learning strategy to use in the model stealing process.
            :param continualLearningStrategy: the continual learning strategy to use in the model stealing process. Note that this class already contains the
            substitute model.
        '''
        super(AlProcess,self).__init__(activeLearningStrategy,trainingConfig,state_dir)

    def active_learning(self,train_set: Dataset, val_set: Dataset,batch_size:int,num_cycles:int,num_epochs:int,
                        optimizer_config:dict,optimizer_builder: Callable,start_cycle:int=0,labeled_set:List[int]=[],
                        unlabeled_set:List[int]=[],score_list:List[float]=[]) -> List[float]:
        '''
            Runs the classic active learning scenario where a new model is trained in every iteration.
            :param train_set: the dataset that the model will be trained on.
            :param val_set: the dataset used to validate the performance of the model
            :param batch_size: the batch size used to train the model.
            :param num_cycles: The number of cycles used in the active learning process.
        '''
        val_loader = DataLoader(val_set,batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        if start_cycle == 0:
            unlabeled_set = [i for i in range(len(train_set))]
            labeled_set = []
            score_list = []
            for i in range(self.al_strat.INIT_BUDGET):
                labeled_set.append(random.randint(0,len(train_set)-1))
            self._train_cycle(train_set,labeled_set,loaders_dict,batch_size,num_epochs,score_list)
            unlabeled_set = [i for i in unlabeled_set if i not in labeled_set]
        
        for i in range(start_cycle,num_cycles):
            self.al_strat.feed_current_state(i,unlabeled_set,labeled_set)
            self.al_strat.model = self.cl_strat.model
            print(f'Running cycle {i+1}/{num_cycles}')
            training_examples = self.al_strat.query()
            training_examples_absolute_indices = [unlabeled_set[elem] for elem in training_examples[-self.al_strat.BUDGET:]]
            labeled_set += training_examples_absolute_indices
            unlabeled_set = [i for i in unlabeled_set if i not in training_examples_absolute_indices]
            optim,scheduler = optimizer_builder(optimizer_config,self.cl_strat.model)
            self.cl_strat.optim = optim
            self.cl_strat.scheduler = scheduler
            self._train_cycle(train_set,labeled_set,loaders_dict,batch_size,num_epochs,score_list)
            if self.state_dir is not None:
                state_dict =  {'start_cycle': i+1, 'labeled_set': labeled_set, 'unlabeled_set': unlabeled_set,'score_list': score_list}
                self._save_state(state_dict)
        return score_list



    def _get_dist(self,data: List[int], num_classes: int) -> List[int]:
        '''
            Returns the distribution of classes in an active learning query.
        '''
        if isinstance(data[0],torch.Tensor):
            data = [elem.item() for elem in data]
        cur_dist = [0] * num_classes
        c = Counter(data)
        for elem in c:
            cur_dist[elem] = round(100*c[elem]/len(data),2)
        return cur_dist
