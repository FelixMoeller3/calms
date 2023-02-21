from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from typing import List
import time
from tqdm import tqdm

class ContinualLearningStrategy(ABC):
    '''
        This is the base class for all continual learning strategies
    '''
    def __init__(self, model:nn.Module,optim: torch.optim.Optimizer,crit: nn.CrossEntropyLoss, use_gpu:bool):
        self.model = model
        self.optim = optim
        self.crit = crit
        self.use_gpu = use_gpu

    def train(self,dataloaders: dict[str,DataLoader],num_epochs:int,val_step:int,result_list:List[float]=[]) -> None:
        start_time = time.time()
        self._before_train()
        for epoch in range(num_epochs):
            print(f'Running epoch {epoch+1}/{num_epochs}')
            self._run_train_epoch(dataloaders["train"])
            log_list = None if epoch < num_epochs-1 else result_list
            if (epoch+1) % val_step == 0:
                self._run_val_epoch(dataloaders['val'],log_list)
        self._after_train(dataloaders['train'].dataset)
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def _before_train(self) -> None:
        pass

    def _run_train_epoch(self,train_loader: DataLoader) -> None:
        '''
            Runs one epoch of the training procedure with the data given by the dataloader.
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data in tqdm(train_loader):

            inputs, labels = data
            if self.use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            self.optim.zero_grad()
            outputs = self.model(inputs)
            # Stop updating the regularization params during training
            #self._update_reg_params(outputs,labels.size(0))
            _, preds = torch.max(outputs.data, 1)
            loss = self.crit(outputs, labels) + self._compute_regularization_loss()

            loss.backward()
            self.optim.step()
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _run_val_epoch(self,dataloader: DataLoader,log_list:List[float]=None) -> None:
        '''
            Runs one validation epoch using the dataloader which contains the validation data. 
        '''
        total_loss = 0.0
        correct_predictions = 0
        self.model.train(False)
        for data in tqdm(dataloader):
            inputs, labels = data
            if self.use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data,1)
            self._after_pred_val(outputs,labels.size(0))
            loss = self.crit(outputs,labels)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)
        if log_list is not None:
            log_list.append(epoch_acc)
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _after_train(self,train_set: Dataset=None) -> None:
        pass

    def _compute_regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def _after_pred_val(self,outputs:torch.Tensor=None,batch_size:int=None) -> None:
        pass

    def save(self, filename: str):
        torch.save(self.model, filename)

    def load(self, filename: str):
        self.model = torch.load(filename)