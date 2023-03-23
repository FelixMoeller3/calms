from abc import ABC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from typing import List
import torch.optim.lr_scheduler as lr_scheduler
import time
from tqdm import tqdm
from torch.nn.utils import clip_grad

class ContinualLearningStrategy(ABC):
    '''
        This is the base class for all continual learning strategies
    '''
    def __init__(self, model:nn.Module,optim: torch.optim.Optimizer, scheduler:lr_scheduler._LRScheduler, crit: nn.CrossEntropyLoss, use_gpu:bool, clip_grad: float=20.0):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.crit = crit
        self.use_gpu = use_gpu
        self.clip_grad = clip_grad
        self.isActive = True


    def train(self,dataloaders: dict[str,DataLoader],num_epochs:int,val_step:int,result_list:List[float]=[],early_stopping:int=-1) -> None:
        '''
            :param early_stopping: Patience (number of epochs) for early stopping. If <0 then no early stopping is used.
        '''
        start_time = time.time()
        if self.isActive:
            self._before_train()
        val_scores = []
        for epoch in range(num_epochs):
            print(f'Running epoch {epoch+1}/{num_epochs}')
            self._run_train_epoch(dataloaders["train"])
            if early_stopping > -1 or (epoch+1) % val_step == 0:
                val_acc = self._run_val_epoch(dataloaders['val'])
            if self.scheduler:
                self.scheduler.step() 
            if early_stopping > -1:
                val_scores.append(val_acc)
                if self._check_stopping(val_scores,early_stopping):
                    print(f"Stopping training after {epoch+1} epochs")
                    break 
        result_list.append(val_acc)
        if self.isActive:
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
            if self.isActive:
                self._before_training_iteration()
            inputs, labels = data
            if self.use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            self.optim.zero_grad()
            outputs = self.model(inputs)
            # Stop updating the regularization params during training
            #self._update_reg_params(outputs,labels.size(0))
            _, preds = torch.max(outputs.data, 1)
            # The labels might be softmax labels, therefore here the class label is computed
            if labels.dim() > 1:
                _, class_labels = torch.max(labels.data, 1)
            else:
                class_labels = labels
            loss = self.crit(outputs, labels)
            if self.isActive:
                loss +=  self._compute_regularization_loss()

            loss.backward()
            self._after_backward()
            if self.clip_grad > 0:
                clip_grad.clip_grad_norm_(self.model.parameters(),self.clip_grad)
            self.optim.step()
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == class_labels.data).item()
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _run_val_epoch(self,dataloader: DataLoader) -> float:
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
            if self.isActive:
                self._after_pred_val(outputs,labels.size(0))
            loss = self.crit(outputs,labels)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        return epoch_acc

    def _after_train(self,train_set: Dataset=None) -> None:
        pass

    def _compute_regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def _after_pred_val(self,outputs:torch.Tensor=None,batch_size:int=None) -> None:
        pass

    def _check_stopping(self,scores: List[float], patience:int) -> bool:
        if len(scores) < 2+patience:
            return False
        first = scores.pop(0)
        return first < min(scores)

    def save(self, filename: str):
        torch.save(self.model, filename)

    def load(self, filename: str):
        self.model = torch.load(filename)

    def deactivate(self) -> None:
        self.isActive = False

    def activate(self) -> None:
        self.isActive = True

    def _after_backward(self) -> None:
        pass

    def _before_training_iteration(self) -> None:
        pass