import torch
import torch.nn as nn
import torch.optim as optim
import time

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

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer,criterion: torch.nn.CrossEntropyLoss,
                 freeze_layers:List[str]=[],use_gpu:bool=False):
        '''
        Initializeses the Memory Aware Synapses (MAS) class.

        Parameters:
        model (nn.Module): the Neural Network which shall be trained

        optimizer (torch.optim.Optimzer)

        Returns:
        None

        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.freeze_layers = freeze_layers
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        self.__init_regularization_params()

    def __init_regularization_params(self) -> None:
        '''
        Initialize the importance weight omega for each parameter of the nn
        '''
        self.regularization_params = {}
        for name, param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            cur_params = {}
            cur_params[IMPORTANCE_WEIGHT_NAME] = torch.zeros(param.size())
            cur_params[PREVIOUS_PARAMS_NAME] = torch.zeros(param.size())
            self.regularization_params[name] = cur_params

    def train(self, dataloaders: dict[str,torch.utils.data.DataLoader], num_epochs:int) -> None:
        '''
            Trains the model for num_epochs with the data given by the dataloader
        '''
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"Running epoch {epoch}/{num_epochs}")

            self.__run_train_epoch(dataloaders['train'])
            self.__run_val_epoch(dataloaders['val'])

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

            

    def __run_train_epoch(self,dataloader: torch.utils.data.DataLoader, dataset_size:int):
        '''
            Runs one train epoch
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data in dataloader:

            inputs, labels = data

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step(self.regularization_params)
            total_loss += loss.data[0]
            correct_predictions += torch.sum(preds == labels.data)
        
        epoch_loss = total_loss / dataset_size
        epoch_acc = correct_predictions / dataset_size

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def __run_val_epoch(self,dataloader: torch.utils.data.Dataloader, dataset_size: int):
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
    
class MAS_Omega_update(optim.Optimizer):
    '''
        This class will be used to update the omegas (i.e. the parameter importance values)
    '''
    def __init__(self, params, lr:float=0.001):
        super(MAS_Omega_update, self).__init__(params,lr)
        
    def __setstate__(self, state: dict) -> None:
        return super(MAS_Omega_update, self).__setstate__(state)

    def step(self,reg_params:dict, batch_index:int, batch_size:int):
        '''
            Updates the importance weights of all parameters
        '''
        for group in self.param_groups:
            for p in group['params']:
                if not p.grad or p not in reg_params:
                    continue
                reg_params[p] = self.__update_weights(reg_params[p],p,batch_index,batch_size)    


    def __update_weights(self,cur_param: dict[str,torch.Tensor],p:torch.Tensor,batch_index:int, batch_size:int):
        omega = cur_param[IMPORTANCE_WEIGHT_NAME]
        omega = omega.to(self.device)
        prev_size = batch_index*batch_size
        cur_size = (batch_index + 1) * batch_size
        omega.mul_(prev_size)
        omega.add_(p.grad.data)
        omega.div_(cur_size)
        cur_param[IMPORTANCE_WEIGHT_NAME] = omega
        return cur_param

