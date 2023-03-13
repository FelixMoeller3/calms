from typing import Callable, List
import torch.nn as nn
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import Dataset,DataLoader,Subset
import random
from .process_base import BaseProcess

class ModelStealingProcess(BaseProcess):

    def __init__(self,targetModel:nn.Module,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy,state_dir:str=None):
        '''
            :param targetModel: the target model in the model stealing process (i.e. the one that will be stolen). Needs to be pretrained!!
            :param activeLearningStrategy: the active learning strategy to use in the model stealing process.
            :param continualLearningStrategy: the continual learning strategy to use in the model stealing process. Note that this class already contains the
            substitute model.
        '''
        super(ModelStealingProcess,self).__init__(activeLearningStrategy,continualLearningStrategy,state_dir)
        self.target_model = targetModel
        self.substitute_model = self.cl_strat.model

    def steal_model(self,train_set: Dataset,val_set: Dataset,batch_size:int,num_cycles:int,num_epochs:int,use_label:bool=True,
                    start_cycle:int=0,labeled_set:List[int]=[],unlabeled_set:List[int]=[],val_accuracies:List[float]=[],
                    agreements:List[float]=[]) -> tuple[List[float],List[float]]:
        '''
            Implements the actual model stealing process where the target model is iteratively queried and then the substitute model is trained using
            continual learning.
            :param train_set: the dataset that the substitute model will be trained on.
            :param val_set: the dataset used to validate the performance of the substitute model
            :param batch_size: the batch size used to train the model.
            :param num_cycles: The number of cycles used in the active learning process.
            :param num_epochs: The number of epochs to train for in each cycle
            :param use_label: Whether to train with the predicted label of the target model or with its softmax probablities
        '''
        # set all labels to -1 to ensure no labels are given before
        #train_set.targets[train_set.targets > -1] = -1
        val_loader = DataLoader(val_set,batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        if start_cycle == 0:
            unlabeled_set = [i for i in range(len(train_set))]
            labeled_set = []
            val_accuracies = []
            agreements = []
            for i in range(self.al_strat.INIT_BUDGET):
                labeled_set.append(random.randint(0,len(train_set)-1))
            self._add_targets(train_set,labeled_set,use_label)
            self._train_cycle(train_set,labeled_set,loaders_dict,batch_size,num_epochs,val_accuracies)
            agreements.append(self._compute_agreement(loaders_dict['val']))
            unlabeled_set = [i for i in unlabeled_set if i not in labeled_set]
        
        for i in range(start_cycle,num_cycles):
            self.al_strat.feed_current_state(i,unlabeled_set,labeled_set)
            print(f'Running cycle {i+1}/{num_cycles}')
            training_examples = self.al_strat.query()
            training_examples_absolute_indices = [unlabeled_set[elem] for elem in training_examples[-self.al_strat.BUDGET:]]
            self._add_targets(train_set,training_examples_absolute_indices,use_label)
            labeled_set += training_examples_absolute_indices
            unlabeled_set = [i for i in unlabeled_set if i not in training_examples_absolute_indices]
            self._train_cycle(train_set,training_examples_absolute_indices,loaders_dict,batch_size,num_epochs,val_accuracies)
            
            if self.state_dir is not None:
                state_dict = {'start_cycle': i+1, 'labeled_set': labeled_set, 'unlabeled_set': unlabeled_set,'val_accuracies': val_accuracies, 'agreements': agreements}
                self._save_state(state_dict)
        return val_accuracies,agreements
    
    def _add_targets(self, train_set: Dataset,labels_to_add: List[int],use_label:bool=True) -> None:
        '''
            Adds the respective targets to the dataset. The targets are determined by querying the target model.
        '''
        if use_label:
            for index in labels_to_add:
                train_set.targets[index] = torch.max(self.target_model(torch.unsqueeze(train_set[index][0],0)),1)[1]
        else:
            for index in labels_to_add:
                train_set.targets[index] = torch.nn.functional.softmax(self.target_model(torch.unsqueeze(train_set[index][0],0)),dim=0)

    
    def _compute_agreement(self, val_loader: DataLoader) -> None:
        agreed_predictions = 0
        for inputs,_ in val_loader:
            _, target_model_preds = torch.max(self.target_model(inputs).data,1)
            _, substitute_model_preds = torch.max(self.cl_strat.model(inputs).data,1)
            agreed_predictions += torch.sum(target_model_preds == substitute_model_preds).item()
        return agreed_predictions/len(val_loader.dataset)
