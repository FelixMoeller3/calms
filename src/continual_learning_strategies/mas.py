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

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer,criterion: torch.nn.CrossEntropyLoss, freeze_layers:List[str]=[],use_gpu:bool=False):
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

    def train(self, dataloader: torch.utils.data.DataLoader, num_epochs:int) -> None:
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"Running epoch {epoch}/{num_epochs}")

            self.optimizer
            self.model.train(True)

            for data in dataloader:

                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step(self.regularization_params)
        


    
class MAS_Omega_update(optim.SGD):
    def __init__(self, params, lr:float=0.001):
        super(MAS_Omega_update, self).__init__(params,lr)
        
    def __setstate__(self, state: dict) -> None:
        return super(MAS_Omega_update, self).__setstate__(state)

    def step(self,reg_params, batch_index:int, batch_size:int):
        pass
