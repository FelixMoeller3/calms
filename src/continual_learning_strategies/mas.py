import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader

from typing import List

IMPORTANCE_WEIGHT_NAME = 'omega'
PREVIOUS_PARAMS_NAME = 'previous'

class MAS:
    '''
    Mas implementation as proposed by the following paper: 
    https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf,
    Based on the following implementations: 
        - https://github.com/wannabeOG/MAS-PyTorch
        - https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
    '''

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer, weight: float, criterion: torch.nn.CrossEntropyLoss,
                 freeze_layers:List[str]=[],use_gpu:bool=False):
        '''
        Initializes the Memory Aware Synapses (MAS) class.

        Parameters:
        model (nn.Module): the Neural Network which shall be trained

        optimizer (torch.optim.Optimizer)

        Returns:
        None

        '''
        self.model = model
        self.optimizer = optimizer
        self.weight = weight
        self.criterion = criterion
        self.freeze_layers = freeze_layers
        self.regularization_params = {}
        self.prev_params = {}
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self._init_regularization_params()

    def _init_regularization_params(self) -> None:
        '''
        Initialize the importance weight omega for each parameter of the nn
        '''
        for name, param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            self.regularization_params[name] = torch.zeros(param.size())

    def train(self, dataloaders: dict[str,DataLoader], num_epochs:int) -> None:
        '''
            Trains the model for num_epochs with the data given by the dataloader
        '''
        start_time = time.time()
        self._save_prev_params()
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch}/{num_epochs}")

            self._run_train_epoch(dataloaders['train'])
            self._run_val_epoch(dataloaders['val'])
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    def _save_prev_params(self):
        for name, param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            self.regularization_params[name] = param.copy_()

    def _run_train_epoch(self,dataloader: DataLoader, dataset_size:int):
        '''
            Runs one train epoch
            TODO: regularization_params dict has changed. Adapt code to it
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        batch_index = 0
        self._init_regularization_params()
        for data in dataloader:

            inputs, labels = data

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels) + self._compute_reg_loss()

            loss.backward()
            self.optimizer.step(self.regularization_params)
            total_loss += loss.data[0]
            correct_predictions += torch.sum(preds == labels.data)
            self._update_reg_params(outputs,batch_index,labels.size(0))
            batch_index += 1
        
        epoch_loss = total_loss / dataset_size
        epoch_acc = correct_predictions / dataset_size

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _update_reg_params(self, outputs: torch.Tensor, batch_index: int, batch_size: int) -> None:
        '''
            Updates the importance weight of each parameter.
        '''
        output_l2 = nn.MSELoss(reduction='none')
        targets = output_l2(outputs,torch.zeros(outputs.size()))
        targets.backward()
        for name, param in self.model.named_parameters():
            if not param.grad or param not in self.regularization_params:
                continue
            self.regularization_params[name] = self._update_weights(self.regularization_params[name],param,batch_index,batch_size)

    def _update_weights(self,cur_param: torch.Tensor,p:torch.Tensor,batch_index:int, batch_size:int):
        cur_param = cur_param.to(self.device)
        prev_size = batch_index*batch_size
        cur_size = (batch_index + 1) * batch_size
        cur_param.mul_(prev_size)
        cur_param.add_(p.grad.data.abs())
        cur_param.div_(cur_size)
        return cur_param

    def _compute_reg_loss(self) -> float:
        '''
            Computes the regularization loss for one batch
        '''
        reg_loss = 0.0
        for name,param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            diff:torch.Tensor = param - self.prev_params[name] 
            diff.mul_(diff)
            diff.mul_(self.regularization_params[name])
            reg_loss += diff.sum()
        return reg_loss

            

    def _run_val_epoch(self,dataloader: DataLoader, dataset_size: int):
        '''
            Runs one validation epoch using the dataloader which contains 
        '''
        total_loss = 0.0
        correct_predictions = 0
        self.model.train(False)
        for data in dataloader:
            inputs, labels = data

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data,1)
            loss = self.criterion(outputs,labels)

            total_loss += loss.data[0]
            correct_predictions += torch.sum(preds == labels.data)

        epoch_loss = total_loss / dataset_size
        epoch_acc = correct_predictions / dataset_size
        
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
