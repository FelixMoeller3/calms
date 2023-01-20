import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader,Dataset
from .cl_base import ContinualLearningStrategy
import time

class ElasticWeightConsolidation(ContinualLearningStrategy):
    '''
        Implementation of Elastic Weight Consolidation (EWC) as proposed in the following paper:
        https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114
        The code is heavily based on the following implementation:
        https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    '''

    def __init__(self,model:nn.Module,optim: torch.optim.Optimizer,crit: nn.CrossEntropyLoss,weight:int=100000):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = optim

    def train(self, dataloaders: dict[str,DataLoader], dataset_sizes: dict[str, int], num_epochs: int):
        '''
            Trains the model for num_epoch epochs using the dataloaders 'train' and 'val' in the dataloaders dict
            which have the sizes dataset_sizes['train'] and dataset_sizes['val'] respectively.
        '''
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            self._run_train_epoch(dataloaders['train'], dataset_sizes['train'])
            self._run_val_epoch(dataloaders['val'], dataset_sizes['val'])
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    def _run_train_epoch(self,dataloader: DataLoader,dataset_size: int) -> None:
        '''
            Runs one epoch of the training procedure with the data given by the dataloader.
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data in dataloader:

            inputs, labels = data

            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self._compute_consolidation_loss() + self.crit(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        
        epoch_loss = total_loss / dataset_size
        epoch_acc = correct_predictions / dataset_size

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _run_val_epoch(self,dataloader: DataLoader, dataset_size: int):
        '''
            Runs one validation epoch using the dataloader which contains the validation data. 
        '''
        total_loss = 0.0
        correct_predictions = 0
        self.model.train(False)
        for data in dataloader:
            inputs, labels = data

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data,1)
            loss = self.crit(outputs,labels)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()

        epoch_loss = total_loss / dataset_size
        epoch_acc = correct_predictions / dataset_size
        
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _update_mean_params(self):
        '''
            TODO: Add method description
        '''
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, current_ds: Dataset, batch_size:int, num_batch:int):
        '''
            TODO: Add method description
        '''
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            output = F.log_softmax(self.model(input), dim=1)
            log_liklihoods.append(output[:, target])
        # TODO: fix calculation in line below 
        # (see https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks/issues/6)
        # for more details
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset: Dataset, batch_size:int, num_batches:int):
        self._update_fisher_params(dataset, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self):
        '''
            TODO: Add method description
        '''
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (self.weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def forward_backward_update(self, input, target):
        '''
            TODO: Add method description
        '''
        output = self.model(input)
        loss = self._compute_consolidation_loss() + self.crit(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    