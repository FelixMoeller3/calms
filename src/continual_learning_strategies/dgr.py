from .cl_base import ContinualLearningStrategy
from models import WGAN
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils import clip_grad
from typing import List
import time
from tqdm import tqdm
from copy import deepcopy

class DeepGenerativeReplay(ContinualLearningStrategy):
    '''
        Implementation of the paper Continual Learning with Deep Generative Replay (https://arxiv.org/abs/1705.08690).
        This code is based on the following implementation: https://github.com/kuc2477/pytorch-deep-generative-replay
    '''

    def __init__(self,model:nn.Module,optim: torch.optim.Optimizer, scheduler: lr_scheduler._LRScheduler,
                 crit: nn.CrossEntropyLoss,USE_GPU:bool=False,clip_grad: float=2.0,**kwargs):
        super(DeepGenerativeReplay,self).__init__(model,optim,scheduler,crit,USE_GPU,clip_grad)
        self.generator = None
        self.num_gen_epochs = 20

    def _train_generator(self,train_loader: DataLoader) -> None:
        '''
            Trains the generator on the given dataset.
        '''
        prev_generator = self.generator
        self.generator = WGAN(image_size=train_loader.dataset[0][0].shape[1],num_channels=train_loader.dataset[0][0].shape[0],use_gpu=self.use_gpu)
        for i in range(self.num_gen_epochs):
            print(f"Running generator epoch {i+1}/{self.num_gen_epochs}")
            for data,labels in tqdm(train_loader,desc="Training generator"):
                if self.use_gpu:
                    data = data.cuda()
                generated_imgs = None
                if prev_generator is not None:
                    generated_imgs = prev_generator.sample(train_loader.batch_size)
                self.generator.train_a_batch(data,generated_imgs)

    def train(self,dataloaders: dict[str,DataLoader],num_epochs:int,val_step:int,result_list:List[float]=[],early_stopping:int=-1) -> None:
        '''
            :param early_stopping: Patience (number of epochs) for early stopping. If <0 then no early stopping is used.
        '''
        if not self.isActive:
            super(DeepGenerativeReplay,self).train(dataloaders,num_epochs,val_step,result_list,early_stopping)
            return
        start_time = time.time()
        val_scores = []
        prev_solver = deepcopy(self.model)
        self._train_generator(dataloaders['train'])
        for epoch in range(num_epochs):
            print(f'Running epoch {epoch+1}/{num_epochs}')
            self._run_solver_epoch(dataloaders["train"],prev_solver)
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

    def _run_solver_epoch(self,train_loader: DataLoader,prev_solver: nn.Module) -> None:
        '''
            Runs one epoch of the training procedure with the data given by the dataloader.
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data,labels in tqdm(train_loader,desc="Training classifier"):
            if self.use_gpu:
                data = data.cuda()
                labels = labels.cuda()
            self.optim.zero_grad()
            outputs = self.model(data)
            loss = self.crit(outputs,labels)
            _,preds = torch.max(outputs.data,1)
            correct_predictions += torch.sum(preds == labels.data)
            if prev_solver is not None:
                gen_data = self.generator.sample(train_loader.batch_size)
                replay_scores = prev_solver(gen_data)
                _,replay_labels = torch.max(replay_scores.data,1)
                gen_outputs = self.model(gen_data)
                loss += self.crit(gen_outputs,replay_labels)
            loss.backward()
            if self.clip_grad > 0:
                clip_grad.clip_grad_norm_(self.model.parameters(),self.clip_grad)
            self.optim.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
