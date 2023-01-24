import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.training import EWC
from avalanche.benchmarks import dataset_benchmark
from torch import autograd
from torch.utils.data import DataLoader,Dataset
from .cl_base import ContinualLearningStrategy
from torchvision.transforms import Compose
from typing import List


class ElasticWeightConsolidation(ContinualLearningStrategy):
    '''
        Implementation of Elastic Weight Consolidation (EWC) as proposed in the following paper:
        https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114
        The code is heavily based on the following implementation:
        https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks and
        https://github.com/thuyngch/Overcoming-Catastrophic-Forgetting
    '''

    def __init__(self,model:nn.Module,optim: torch.optim.Optimizer,crit: nn.CrossEntropyLoss,weight:int=100000):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = optim
        self.ewc = EWC(self.model,self.optimizer,self.crit,self.weight,'separate')

    def train(self, train_datasets: List[Dataset], test_datasets: List[Dataset], num_epochs: int, batch_size:int=100 ,transform: Compose=None):
        '''
            Trains the model for num_epoch epochs using the dataloaders 'train' and 'val' in the dataloaders dict
        '''
        self.ewc.train_epochs = num_epochs
        self.ewc.train_mb_size = batch_size
        self.ewc.eval_mb_size = batch_size
        scenario = dataset_benchmark(train_datasets,test_datasets,train_transform=transform,eval_transform=transform)
        self.ewc.train(scenario.train_stream)
        self.ewc.eval(scenario.test_stream)
        '''
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            self.ewc.train()
            self._run_train_epoch(dataloaders['train'])
            self._run_val_epoch(dataloaders['val'])
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        '''

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

    def _update_fisher_params(self, dataloader: DataLoader):
        '''
            TODO: Add method description
        '''
        log_likelihoods = []
        for i, (input, target) in enumerate(dataloader):
            input = input.view(input.shape[0],-1)
            if i> 0 and input.shape[0] != log_likelihoods[0].shape[0]:
                break
            output = F.log_softmax(self.model(input), dim=1)
            log_likelihoods.append(output[:, target])
        # TODO: fix calculation in line below 
        # (see https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks/issues/6)
        # for more details
        log_likelihood = torch.cat(log_likelihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataloader: DataLoader):
        self._update_fisher_params(dataloader)
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
    

    def eval(self,evalset: Dataset, transform: Compose=None):
        scenario = dataset_benchmark([],[evalset],train_transform=transform,eval_transform=transform)
        self.ewc.eval(scenario.test_stream[0])