import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader,Dataset
from .cl_base import ContinualLearningStrategy

class ElasticWeightConsolidation(ContinualLearningStrategy):
    '''
        Implementation of Elastic Weight Consolidation (EWC) as proposed in the following paper:
        https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114
        The code is heavily based on the following implementation:
        https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    '''

    def __init__(self,model:nn.Module,crit: nn.CrossEntropyLoss,lr:float=0.001,weight:int=100000):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = optim.SGD(self.model.parameters(),lr)

    def train(self, dataloader: DataLoader, num_epochs: int):
        for _ in range(num_epochs):
            for inputs, labels in dataloader:
                self.model.train(True)
                self.forward_backward_update(inputs, labels)


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
    