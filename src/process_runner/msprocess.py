from typing import Callable, List
import torch.nn as nn
import torch
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import Dataset,DataLoader,Subset
import random
from .process_base import BaseProcess
from data import Softmax_label_set


class ModelStealingProcess(BaseProcess):

    def __init__(self,targetModel:nn.Module,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy,state_dir:str=None,use_gpu:bool=False):
        '''
            :param targetModel: the target model in the model stealing process (i.e. the one that will be stolen). Needs to be pretrained!!
            :param activeLearningStrategy: the active learning strategy to use in the model stealing process.
            :param continualLearningStrategy: the continual learning strategy to use in the model stealing process. Note that this class already contains the
            substitute model.
        '''
        super(ModelStealingProcess,self).__init__(activeLearningStrategy,continualLearningStrategy,state_dir)
        self.target_model = targetModel
        self.substitute_model = self.cl_strat.model
        self.use_gpu = use_gpu

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
        if not use_label:
            train_set = Softmax_label_set(train_set)
        val_loader = DataLoader(val_set,batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        if start_cycle == 0:
            unlabeled_set = [i for i in range(len(train_set))]
            labeled_set = [i for i in range(len(train_set))]
            val_accuracies = []
            agreements = []
            random.shuffle(labeled_set)
            #labeled_set = labeled_set[:self.al_strat.INIT_BUDGET]
            labeled_set = [i for i in range(0,1000)]
            self._add_targets(train_set,labeled_set,use_label)
            print(train_set.targets[0])
            print(train_set.targets[2])
            print(train_set[0])
            print(train_set[2])
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
            cur_agreement = self._compute_agreement(loaders_dict['val'])
            print("Current agreement is: {:.4f}".format(cur_agreement))
            agreements.append(cur_agreement)
            if self.state_dir is not None:
                state_dict = {'start_cycle': i+1, 'labeled_set': labeled_set, 'unlabeled_set': unlabeled_set,'val_accuracies': val_accuracies, 'agreements': agreements}
                self._save_state(state_dict)
        return val_accuracies,agreements
    
    def _add_targets(self, train_set: Dataset,labels_to_add: List[int],use_label:bool) -> None:
        '''
            Adds the respective targets to the dataset. The targets are determined by querying the target model.
        '''
        # TODO: Adapt to gpu usage
        with torch.no_grad():
            if use_label:
                for index in labels_to_add:
                    cur_elem = train_set[index][0].cuda() if self.use_gpu else train_set[index][0]
                    a,b = torch.max(self.target_model(torch.unsqueeze(cur_elem,0)),1)
                    if b.item() > 9:
                        print("Error")
                    train_set.targets[index] = b.item()
            else:
                for index in labels_to_add:
                    cur_elem = train_set[index][0].cuda() if self.use_gpu else train_set[index][0]
                    current_example = torch.unsqueeze(cur_elem,0)
                    output = torch.squeeze(self.target_model(current_example))
                    softmax_res = torch.nn.functional.softmax(output,dim=0)
                    train_set.targets[index] = softmax_res
    
    def _compute_agreement(self, val_loader: DataLoader) -> None:
        agreed_predictions = 0
        for inputs,_ in val_loader:
            inputs = inputs.cuda() if self.use_gpu else inputs
            _, target_model_preds = torch.max(self.target_model(inputs).data,1)
            _, substitute_model_preds = torch.max(self.cl_strat.model(inputs).data,1)
            agreed_predictions += torch.sum(target_model_preds == substitute_model_preds).item()
        return agreed_predictions/len(val_loader.dataset)
