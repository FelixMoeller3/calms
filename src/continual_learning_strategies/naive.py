from .cl_base import ContinualLearningStrategy
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Naive(ContinualLearningStrategy):

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer,criterion: torch.nn.CrossEntropyLoss,use_gpu:bool=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self,train_loader: DataLoader,num_epochs:int) -> None:
        self.model.train(True)
        for i in range(num_epochs):
            print(f'Running epoch {i+1}/{num_epochs}')
            self._run_train_epoch(train_loader)

    def _run_train_epoch(self,train_loader: DataLoader) -> None:
        total_loss = 0.0
        correct_predictions = 0
        for data in tqdm(train_loader):

            inputs, labels = data

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # Stop updating the regularization params during training
            #self._update_reg_params(outputs,labels.size(0))
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def eval(self,eval_loader: DataLoader) -> None:
        self.model.train(False)
        total_loss = 0.0
        correct_predictions = 0
        for data in tqdm(eval_loader):

            inputs, labels = data

            outputs = self.model(inputs)
            # Stop updating the regularization params during training
            #self._update_reg_params(outputs,labels.size(0))
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        epoch_loss = total_loss / len(eval_loader.dataset)
        epoch_acc = correct_predictions / len(eval_loader.dataset)

        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

