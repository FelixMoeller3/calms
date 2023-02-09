import torch
from .cl_base import ContinualLearningStrategy
from torch import nn
from torch.utils.data import DataLoader,Dataset
import time
from tqdm import tqdm
from typing import List, Optional
import random
from torch.nn import functional as F

class IMM(ContinualLearningStrategy):
    '''
    Implementation of Incremental Moment Matching (IMM) according.
    '''

    def __init__(self, model: nn.Module,optim: torch.optim.Optimizer,crit: nn.CrossEntropyLoss,alphas:List[float]=None,weight:float=1.0,mean:bool=True):
        '''
            :param alphas: List of weights for models of previous tasks,
             i.e. how strongly previous tasks should be weighted. The sum of all entries in this list must be 1.
             If no list is provided the all previous tasks will be weighted equally.
            :param mean: Whether to use mean-IMM or mode-IMM
        '''
        self.model = model
        self.optimizer = optim
        self.crit = crit
        assert not alphas or abs(sum(alphas)-1.0) < 1e-8
        self.alphas = alphas
        self.weight = weight
        self.mean = mean
        self.num_tasks = 0
        self.prev_param_list = []
        self._save_model_params()
        self.prev_fishers = []

    def _save_model_params(self) -> None:
        prev_params = {}
        for name, param in self.model.named_parameters():
            prev_params[name] = param.detach().clone()
        self.prev_param_list.append(prev_params)
        # delete the first model in the list if its alpha value would be 0
        # because the list of previous models is longer than the list of alpha values
        if self.alphas and len(self.prev_param_list) > len(self.alphas):
            self.prev_param_list.pop(0)

    def train(self, dataloaders: dict[str,DataLoader], num_epochs: int) -> None:
        '''
            Trains the model for num_epoch epochs using the dataloaders 'train' and 'val' in the dataloaders dict
        '''
        start_time = time.time()
        self._set_model_params()
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            self._run_train_epoch(dataloaders['train'])
            self._run_val_epoch(dataloaders['val'])
        self._save_model_params()
        self._merge_models(dataloaders['train'].dataset)
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def _set_model_params(self) -> None:
        for name,param in self.model.named_parameters():
            param.data = self.prev_param_list[-1][name]

    def _run_train_epoch(self,dataloader: DataLoader) -> None:
        '''
            Runs one epoch of the training procedure with the data given by the dataloader.
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data in tqdm(dataloader):

            inputs, labels = data

            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self._compute_consolidation_loss() + self.crit(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    
    def _run_val_epoch(self,dataloader: DataLoader) -> None:
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
        
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _compute_consolidation_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name,param in self.model.named_parameters():
            diff:torch.Tensor = param - self.prev_param_list[-1][name]
            diff.pow_(2)
            loss += diff.sum()
        return loss * self.weight

    def _merge_models(self, train_dataset:Optional[Dataset], sample_size:Optional[float]=0.05) -> None:
        new_model_weights = {}
        for name,param in self.model.named_parameters():
            new_model_weights[name] = torch.zeros_like(param)
        alphas = [1/len(self.prev_param_list)] * len(self.prev_param_list) if not self.alphas else self.alphas
        if self.mean:
            for i,weights in enumerate(self.prev_param_list):
                for name,param in self.model.named_parameters():
                    new_model_weights[name] += alphas[i] * weights[name]

        else:
            '''
                TODO: The numbers seem odd for mode-IMM. Check if calculation works 
            '''
            self._calc_fisher(train_dataset,sample_size)
            sigma = self._calc_sigma()
            for i,(weights,fisher) in enumerate(zip(self.prev_param_list,self.prev_fishers)):
                for name,param in self.model.named_parameters():
                    new_model_weights[name] += alphas[i] * weights[name] * fisher[name]
            for param_name in new_model_weights:
                new_model_weights[param_name] *= sigma[param_name]
                
        for name,param in self.model.named_parameters():
            param.data = new_model_weights[name]

        
    def _calc_fisher(self,train_dataset: Dataset=None, sample_size:float=0.05) -> None:
        num_samples = (len(train_dataset) * sample_size) if train_dataset else 0
        fisher = {}
        for name,param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        for _ in range(int(num_samples)):
            cur_index = random.randint(0,len(train_dataset)-1)
            elem, label = train_dataset[cur_index]
            self.optimizer.zero_grad()
            output = self.model(elem)
            sm = F.log_softmax(output,dim=1)
            label_tensor = torch.tensor([label],dtype=torch.long)
            loss = F.nll_loss(sm,label_tensor)
            loss.backward()

            for name, param in self.model.named_parameters():
                fisher[name] += param.grad.data ** 2 / num_samples
                # add small epsilon to fisher values to make sure the are greater than zero.
                # This is necessary because the values will be inverted later
                fisher[name] += 1e-8

        self.prev_fishers.append(fisher)
        # delete the first fisher matrix in the list if its alpha value would be 0
        # because the list of previous fisher matrices is longer than the list of alpha values
        if self.alphas and len(self.prev_fishers) > len(self.alphas):
            self.prev_fishers.pop(0)
        

    def _calc_sigma(self) -> dict[str,torch.Tensor]:
        sigma = {}
        for name,param in self.model.named_parameters():
            sigma[name] = torch.zeros_like(param)
        alphas =  [1/len(self.prev_param_list)] * len(self.prev_param_list) if not self.alphas else self.alphas
        for name,param in self.model.named_parameters():
            for i,weights in enumerate(self.prev_fishers):
                sigma[name] += alphas[i-1] * weights[name]
                sigma[name] = 1/sigma[name]

        return sigma

