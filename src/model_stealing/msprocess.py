from typing import Callable, List
import torch.nn as nn
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import Dataset,DataLoader,Subset
import random
from models import testConv,ResNet,BasicBlock
from collections import Counter
import copy

class ModelStealingProcess:

    def __init__(self,targetModel:nn.Module,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy):
        '''
            :param targetModel: the target model in the model stealing process (i.e. the one that will be stolen). Needs to be pretrained!!
            :param activeLearningStrategy: the active learning strategy to use in the model stealing process.
            :param continualLearningStrategy: the continual learning strategy to use in the model stealing process. Note that this class already contains the
            substitute model.
        '''
        self.target_model = targetModel
        self.al_strat = activeLearningStrategy
        self.cl_strat = continualLearningStrategy
        self.substitute_model = self.cl_strat.model

    def active_learning(self,train_set: Dataset, val_set: Dataset,batch_size:int,num_cycles:int,num_epochs:int,
                        optimizer_config:dict,optimizer_builder: Callable) -> List[float]:
        '''
            Runs the classic active learning scenario where a new model is trained in every iteration.
            :param train_set: the dataset that the model will be trained on.
            :param val_set: the dataset used to validate the performance of the model
            :param batch_size: the batch size used to train the model.
            :param num_cycles: The number of cycles used in the active learning process.
        '''
        val_loader = DataLoader(val_set,batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        unlabeled_set = [i for i in range(len(train_set))]
        labeled_set = []
        score_list = []
        optimizer_state = self.cl_strat.optim.state_dict()
        copied_scheduler = self.cl_strat.scheduler.state_dict
        for i in range(self.al_strat.INIT_BUDGET):
            labeled_set.append(random.randint(0,len(train_set)-1))
        training_set = Subset(train_set,labeled_set)
        loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
        self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list,5)
        unlabeled_set = [i for i in unlabeled_set if i not in labeled_set]
        for i in range(num_cycles):
            self.al_strat.feed_current_state(i,unlabeled_set,labeled_set)
            self.al_strat.model = self.cl_strat.model
            print(f'Running cycle {i+1}/{num_cycles}')
            training_examples = self.al_strat.query()
            labeled_set += list(training_examples)
            unlabeled_set = [i for i in unlabeled_set if i not in training_examples]
            training_set = Subset(train_set,labeled_set)
            loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
            self.cl_strat.model = ResNet(BasicBlock, [2,2,2,2], 10)
            self.cl_strat.model = self.cl_strat.model.cuda()
            optim,scheduler = optimizer_builder(optimizer_config,self.cl_strat.model)
            self.cl_strat.optim = optim
            self.cl_strat.scheduler = scheduler
            self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list,5)

        return score_list

    def continual_learning(self,train_set: Dataset, val_set: Dataset,batch_size:int,num_cycles:int,
    num_epochs:int,compute_query_dist:bool=False) -> tuple[List[float],List[List[float]]]:
        '''
            Runs a combined continual and active learning approach where instead of querying the target model the actual label is used.
            :param train_set: the dataset that the model will be trained on.
            :param val_set: the dataset used to validate the performance of the model
            :param batch_size: the batch size used to train the model.
            :param num_cycles: The number of cycles used in the active learning process.
        '''
        val_loader = DataLoader(val_set,batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        unlabeled_set = [i for i in range(len(train_set))]
        labeled_set = []
        score_list = []
        dist_list = []
        for i in range(self.al_strat.INIT_BUDGET):
            labeled_set.append(random.randint(0,len(train_set)-1))
        if compute_query_dist:
            labels = [train_set.targets[index] for index in labeled_set]
            dist_list.append(self._get_dist(labels,len(train_set.class_to_idx)))
        training_set = Subset(train_set,labeled_set)
        loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
        self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list,5)
        unlabeled_set = [i for i in unlabeled_set if i not in labeled_set]
        for i in range(num_cycles):
            self.al_strat.feed_current_state(i,unlabeled_set,labeled_set)
            print(f'Running cycle {i+1}/{num_cycles}')
            training_examples = self.al_strat.query()
            if compute_query_dist:
                labels = [train_set.targets[index] for index in training_examples[-self.al_strat.BUDGET:]]
                dist_list.append(self._get_dist(labels,len(train_set.class_to_idx)))
            labeled_set += list(training_examples[-self.al_strat.BUDGET:])
            unlabeled_set = [i for i in unlabeled_set if i not in training_examples[-self.al_strat.BUDGET:]]
            training_set = Subset(train_set,training_examples)
            loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
            self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list,5)

        return score_list,dist_list

    def steal_model(self,train_set: Dataset,val_set: Dataset,batch_size:int,num_cycles:int,num_epochs:int) -> List[float]:
        '''
            Implements the actual model stealing process where the target model is iteratively queried and then the substitute model is trained using
            continual learning.
            :param train_set: the dataset that the substitute model will be trained on.
            :param val_set: the dataset used to validate the performance of the substitute model
            :param batch_size: the batch size used to train the model.
            :param num_cycles: The number of cycles used in the active learning process.
        '''
        # set all labels to -1 to ensure no labels are given before
        train_set.targets[train_set.targets > -1] = -1
        val_loader = DataLoader(val_set,batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        unlabeled_set = [i for i in range(len(train_set))]
        labeled_set = []
        score_list = []
        for i in range(self.al_strat.INIT_BUDGET):
            labeled_set.append(random.randint(0,len(train_set)-1))
        for elem in labeled_set:
            train_set.targets[elem] = torch.max(self.target_model(torch.unsqueeze(train_set[elem][0],0)),1)[1]
            #print(train_set.targets[elem])
        training_set = Subset(train_set,labeled_set)
        loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
        self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list,5)
        unlabeled_set = [i for i in unlabeled_set if i not in labeled_set]
        for i in range(num_cycles):
            self.al_strat.feed_current_state(i,unlabeled_set,labeled_set)
            print(f'Running cycle {i+1}/{num_cycles}')
            training_examples = self.al_strat.query()
            for elem in training_examples:
                train_set.targets[elem] = torch.max(self.target_model(torch.unsqueeze(train_set[elem][0],0)),1)[1]
            labeled_set += list(training_examples[-self.al_strat.BUDGET:])
            unlabeled_set = [i for i in unlabeled_set if i not in training_examples[-self.al_strat.BUDGET:]]
            training_set = Subset(train_set,training_examples)
            loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
            self.cl_strat.train(loaders_dict,num_epochs,num_epochs,score_list,5)

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
            cur_dist[elem] = c[elem]/len(data)
        return cur_dist
