import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from typing import List, Union
from .cl_base import ContinualLearningStrategy

class Weight_Regularized_SGD(optim.SGD):
    '''
    Implements SGD training with importance params regularization.
    '''
    def __init__(self, params, lr:float=0.001, momentum:float=0, dampening:float=0,
                 weight_decay:float=0, nesterov:bool=False,weight:float=1.0):
        self.weight = weight
        super(Weight_Regularized_SGD, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)

    def step(self, reg_params:dict[str,Parameter],prev_params: dict[str,Parameter]):
        """Performs a single optimization step.
        Arguments:
            reg_params: omega of all the params
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
           
            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad.data
               
                #MAS PART CODE GOES HERE
                #if this param has an omega to use for regulization
                if param in reg_params:
                    d_p = self._update_gradient(param,d_p,reg_params[param],prev_params[param],self.weight)
                #MAS PARAT CODE ENDS
                if momentum != 0:
                    d_p = self._add_momentum(param,d_p,self.state[param],momentum,dampening)
                param.data.add_(-group['lr'], d_p)


    def _update_gradient(self,param: Parameter,grad: torch.Tensor,reg_param: torch.Tensor,prev_param: torch.Tensor,reg_lambda:float):
        #initial value when the training start
        curr_weight_val=param.data
        #get the difference
        weight_dif=curr_weight_val.add(-1,prev_param)
        #compute the MAS penalty
        regulizer=weight_dif.mul(2*reg_lambda*reg_param)
        #add the MAS regulizer to the gradient
        grad.add_(regulizer)
        return grad

    def _add_momentum(self, param, grad, param_state, momentum: float, dampening: float):
        param_state = self.state[param]
        if 'momentum_buffer' not in param_state:
            param_state['momentum_buffer'] = grad.clone()
            buf = param_state['momentum_buffer']
        else:
            buf = param_state['momentum_buffer']
            buf.mul_(momentum).add_(1 - dampening, grad)
        return buf

class MAS(ContinualLearningStrategy):
    '''
    Mas implementation as proposed by the following paper: 
    https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf,
    Based on the following implementations: 
        - https://github.com/wannabeOG/MAS-PyTorch
        - https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
    '''

    def __init__(self,model:nn.Module,optimizer: Weight_Regularized_SGD, weight: float, criterion: torch.nn.CrossEntropyLoss,
                 freeze_layers:List[str]=[],lr:float=0.001,momentum:float=0,use_gpu:bool=False):
        '''
        Initializes the Memory Aware Synapses (MAS) class.

        Parameters:
        model (nn.Module): the Neural Network which shall be trained

        optimizer (torch.optim.Optimizer)

        Returns:
        None

        '''
        self.model = model
        self.optimizer = Weight_Regularized_SGD(model.parameters(),lr,momentum,weight)
        self.weight = weight
        self.criterion = criterion
        self.freeze_layers = freeze_layers
        # The total number of samples that have been classified before training the current task
        self.n_samples_prev = 0
        self.regularization_params_prev = {}
        # The number of samples from the current task that is learned that have been classified until now
        self.n_samples_cur = 0
        self.regularization_params_cur = {}
        
        self.prev_params = {}
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self._init_regularization_params()

    def _init_regularization_params(self) -> None:
        '''
            Initialize the importance weight omega for each parameter of the nn
        '''
        if self.n_samples_prev == 0:
            for name, param in self.model.named_parameters():
                if name in self.freeze_layers:
                    continue
                self.regularization_params_prev[name] = torch.zeros(param.size())
        # return if the current parameters have already been set to zero
        for name, param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            self.regularization_params_cur[name] = torch.zeros(param.size())

    def train(self, dataloaders: dict[str,DataLoader], num_epochs:int) -> None:
        '''
            Trains the model for num_epochs with the data given by the dataloader
        '''
        start_time = time.time()
        self._save_prev_params()
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch+1}/{num_epochs}")

            self._run_train_epoch(dataloaders['train'])
            self._run_val_epoch(dataloaders['val'])
        self._update_regularization_params()
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    def _save_prev_params(self) -> None:
        '''
            Saves the parameters of the model before training the next task
        '''
        for name, param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            self.prev_params[name] = param.data.detach().clone()

    def _update_regularization_params(self) -> None:
        '''
            Updates the regularization omegas with the information of the task that was just learned.
            Should be called after a new task is trained.
        '''
        total = self.n_samples_prev + self.n_samples_cur
        for name, _ in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            prev_total = self.regularization_params_prev[name] * self.n_samples_prev
            cur_total = self.regularization_params_cur[name] * self.n_samples_cur
            self.regularization_params_prev[name] = (prev_total + cur_total) / total
        self.n_samples_prev += self.n_samples_cur
        self.n_samples_cur = 0
        self._init_regularization_params()


    def _run_train_epoch(self,dataloader: DataLoader):
        '''
            Runs one train epoch
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data in dataloader:

            inputs, labels = data

            self.optimizer.zero_grad()
            self.model.zero_grad()
            outputs = self.model(inputs)
            # Stop updating the regularization params during training
            #self._update_reg_params(outputs,labels.size(0))
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step(self.regularization_params_prev,self.prev_params)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _update_reg_params(self, outputs: torch.Tensor, batch_size: int) -> None:
        '''
            Updates the importance weight omega of each parameter. 
            Should be called whenever the model is evaluated
        '''
        output_l2 = nn.MSELoss(reduction='sum')
        targets = output_l2(outputs,torch.zeros(outputs.size()))
        targets.backward(retain_graph=True)
        for name, param in self.model.named_parameters():
            if name not in self.regularization_params_prev:
                continue
            self.regularization_params_cur[name] = self._update_weights(self.regularization_params_cur[name],param,batch_size)

    def _update_weights(self,cur_param: torch.Tensor,p:torch.Tensor, batch_size:int):
        '''
            Updates the importance weight omega for a given parameter.
        '''
        cur_param = cur_param.to(self.device)
        prev_size = self.n_samples_cur
        self.n_samples_cur += batch_size
        cur_param.mul_(prev_size)
        gradient = p.grad.data.clone()
        cur_param.add_(gradient.abs())
        cur_param.div_(self.n_samples_cur)
        return cur_param

    def _compute_reg_loss(self) -> float:
        '''
            Computes the regularization loss for one batch
        '''
        reg_loss = torch.tensor(0.0)
        for name,param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            newP = param.detach().clone()
            diff = newP - self.prev_params[name] 
            diff.mul_(diff)
            diff.mul_(self.regularization_params_prev[name])
            reg_loss += diff.sum()
        return self.weight * reg_loss

            

    def _run_val_epoch(self,dataloader: DataLoader):
        '''
            Runs one validation epoch using the dataloader which contains the validation set and has dataset_size elements
        '''
        total_loss = 0.0
        correct_predictions = 0
        self.model.train(False)
        for data in dataloader:
            inputs, labels = data
            self.model.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data,1)
            loss = self.criterion(outputs,labels)
            batch_size = labels.size(0)
            self._update_reg_params(outputs,batch_size)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)
        
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
