from typing import Callable, List
import torch.nn as nn
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import Dataset,DataLoader
from .process_base import BaseProcess
from data import Softmax_label_set


class ModelStealingProcess(BaseProcess):

    def __init__(self,targetModel:nn.Module,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy,train_set: Dataset,
            val_set: Dataset,batch_size:int,num_cycles:int,num_epochs:int,continual:int,num_classes:int,optimizer_builder: Callable,optimizer_config:dict,
                 use_label:bool=True,state_dir:str=None,use_gpu:bool=False):
        '''
            :param targetModel: the target model in the model stealing process (i.e. the one that will be stolen). Needs to be pretrained!!
            :param activeLearningStrategy: the active learning strategy to use in the model stealing process.
            :param continualLearningStrategy: the continual learning strategy to use in the model stealing process. Note that this class already contains the
            substitute model.
            :param train_set: the dataset that the substitute model will be trained on.
            :param val_set: the dataset used to validate the performance of the substitute model
            :param batch_size: the batch size used to train the model.
            :param num_cycles: The number of cycles used in the active learning process.
            :param num_epochs: The number of epochs to train for in each cycle
            :param use_label: Whether to train with the predicted label of the target model or with its softmax probablities
        '''
        super(ModelStealingProcess,self).__init__(activeLearningStrategy,continualLearningStrategy,train_set,val_set,batch_size,num_cycles,num_epochs,
                                                  continual,optimizer_builder,optimizer_config,use_gpu,state_dir)
        self.target_model = targetModel
        self.num_classes = num_classes
        self.substitute_model = self.cl_strat.model
        self.use_label = use_label

    def steal_model(self,start_cycle:int=0,labeled_set:List[int]=[],unlabeled_set:List[int]=[],val_accuracies:List[float]=[],
                    agreements:List[float]=[]) -> tuple[List[float],List[float]]:
        '''
            Implements the actual model stealing process where the target model is iteratively queried and then the substitute model is trained using
            continual learning.
        '''
        # set all labels to -1 to ensure no labels are given before
        #train_set.targets[train_set.targets > -1] = -1
        if not self.use_label:
            self.train_set = Softmax_label_set(self.train_set,self.num_classes)
        val_loader = DataLoader(self.val_set,self.batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        if start_cycle == 0:
            labeled_set,unlabeled_set = self._before_first_cycle(loaders_dict,val_accuracies)
            agreements = []
            cur_agreement = self._compute_agreement(loaders_dict['val'])
            print("Current agreement is: {:.4f}".format(cur_agreement))
            agreements.append(cur_agreement)
        
        for i in range(start_cycle,self.num_cycles):
            print(f'Running cycle {i+1}/{self.num_cycles}')
            if i == self.continualStart:
                print("Switching from pure active learning to continual active learning")
            labeled_set,unlabeled_set = self._query_cycle(i,labeled_set,unlabeled_set,loaders_dict,val_accuracies)
            cur_agreement = self._compute_agreement(loaders_dict['val'])
            print("Current agreement is: {:.4f}".format(cur_agreement))
            agreements.append(cur_agreement)
            if self.state_dir is not None:
                state_dict = {'start_cycle': i+1, 'labeled_set': labeled_set, 'unlabeled_set': unlabeled_set,'val_accuracies': val_accuracies, 'agreements': agreements}
                self._save_state(state_dict)
        return val_accuracies,agreements
    
    def _add_targets(self,labels_to_add: List[int]) -> None:
        '''
            Adds the respective targets to the dataset. The targets are determined by querying the target model.
        '''
        with torch.no_grad():
            if self.use_label:
                for index in labels_to_add:
                    cur_elem = self.train_set[index][0].cuda() if self.use_gpu else self.train_set[index][0]
                    self.train_set.targets[index] = torch.max(self.target_model(torch.unsqueeze(cur_elem,0)),1)[1].item()
            else:
                for index in labels_to_add:
                    cur_elem = self.train_set[index][0].cuda() if self.use_gpu else self.train_set[index][0]
                    current_example = torch.unsqueeze(cur_elem,0)
                    output = torch.squeeze(self.target_model(current_example))
                    softmax_res = torch.nn.functional.softmax(output,dim=0)
                    self.train_set.targets[index] = softmax_res
    
    def _compute_agreement(self, val_loader: DataLoader) -> None:
        agreed_predictions = 0
        for inputs,_ in val_loader:
            inputs = inputs.cuda() if self.use_gpu else inputs
            _, target_model_preds = torch.max(self.target_model(inputs).data,1)
            _, substitute_model_preds = torch.max(self.substitute_model(inputs).data,1)
            agreed_predictions += torch.sum(target_model_preds == substitute_model_preds).item()
        return agreed_predictions/len(val_loader.dataset)
