from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from .cl_base import ContinualLearningStrategy
import time
import random
from torch.nn import functional as F
from tqdm import tqdm


class ElasticWeightConsolidation(ContinualLearningStrategy):
    '''
        Implementation of Elastic Weight Consolidation (EWC) as proposed in the following paper:
        https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114
        The code is heavily based on the following implementations:
        https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks and
        https://github.com/thuyngch/Overcoming-Catastrophic-Forgetting
    '''

    def __init__(self,model:nn.Module,optim: torch.optim.Optimizer,crit: nn.CrossEntropyLoss,WEIGHT:float=1.0,**kwargs):
        super(ElasticWeightConsolidation,self).__init__(model,optim,crit)
        self.weight = WEIGHT
        self.prev_params = {}
        self._save_model_params()
        self.fisher = {}
        self._update_fisher_params()

    def _save_model_params(self) -> None:
        '''
            Saves the current parameters of the model before training on a new task because the old 
            model parameters are needed to compute the loss function
        '''
        for name,param in self.model.named_parameters():
            self.prev_params[name] = param.detach().clone()

    def train(self, dataloaders: dict[str,DataLoader], num_epochs: int,result_list:List[float]=[]):
        '''
            Trains the model for num_epoch epochs using the dataloaders 'train' and 'val' in the dataloaders dict
        '''
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            self._run_train_epoch(dataloaders['train'])
            log_list = None if epoch < num_epochs-1 else result_list
            self._run_val_epoch(dataloaders['val'],log_list)
        self._save_model_params()
        self._update_fisher_params(dataloaders['train'].dataset,0.05)
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def _run_train_epoch(self,dataloader: DataLoader) -> None:
        '''
            Runs one epoch of the training procedure with the data given by the dataloader.
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data in tqdm(dataloader):

            inputs, labels = data

            self.optim.zero_grad()
            
            outputs = self.model(inputs)
            loss = self._compute_consolidation_loss() + self.crit(outputs, labels)
            loss.backward()
            self.optim.step()
            _, preds = torch.max(outputs.data, 1)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _run_val_epoch(self,dataloader: DataLoader,log_list:List[float]=None):
        '''
            Runs one validation epoch using the dataloader which contains the validation data. 
        '''
        total_loss = 0.0
        correct_predictions = 0
        self.model.train(False)
        for data in tqdm(dataloader):
            inputs, labels = data

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data,1)
            loss = self.crit(outputs,labels)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)
        if log_list is not None:
            log_list.append(epoch_acc)
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


    def _update_fisher_params(self, train_dataset: Dataset=None, sample_size:float=0.05):
        '''
            TODO: Add method description
        '''
        num_samples = (len(train_dataset) * sample_size) if train_dataset else 0
        self.fisher = {}
        for name,param in self.model.named_parameters():
            self.fisher[name] = torch.zeros_like(param)
        for _ in range(int(num_samples)):
            cur_index = random.randint(0,len(train_dataset)-1)
            elem, label = train_dataset[cur_index]
            self.optim.zero_grad()
            input = torch.unsqueeze(elem,0)
            output = self.model(input)
            sm = F.log_softmax(output,dim=1)
            label_tensor = torch.tensor([label],dtype=torch.long)
            loss = F.nll_loss(sm,label_tensor)
            loss.backward()

            for name, param in self.model.named_parameters():
                self.fisher[name] += param.grad.data ** 2 / num_samples


    def _compute_consolidation_loss(self):
        '''
            TODO: Add method description
        '''
        loss = torch.tensor(0.0)
        for name,param in self.model.named_parameters():
            diff:torch.Tensor = param - self.prev_params[name]
            diff.pow_(2)
            diff.mul_(self.fisher[name])
            loss += diff.sum()
        return loss * (self.weight / 2)
        