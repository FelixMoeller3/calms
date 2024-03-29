import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List
from .cl_base import ContinualLearningStrategy
import torch.optim.lr_scheduler as lr_scheduler

class MAS(ContinualLearningStrategy):
    '''
    Mas implementation as proposed by the following paper: 
    https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf,
    Based on the following implementations: 
        - https://github.com/wannabeOG/MAS-PyTorch
        - https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
    '''

    def __init__(self,model:nn.Module,optimizer: torch.optim.Optimizer, scheduler: lr_scheduler._LRScheduler,criterion: torch.nn.CrossEntropyLoss,
            WEIGHT:float=1.0,SCHEDULE:bool=False,FREEZE_LAYERS:List[str]=[],USE_GPU:bool=False,clip_grad:float=2.0,state_dict:dict=None,**kwargs):
        '''
        Initializes the Memory Aware Synapses (MAS) class.

        Parameters:
        model (nn.Module): the Neural Network which shall be trained

        optimizer (torch.optim.Optimizer)

        Returns:
        None

        '''
        super(MAS,self).__init__(model,optimizer,scheduler,criterion,USE_GPU,clip_grad)
        if state_dict is not None:
            self.set_state(state_dict)
            return
        self.weight = torch.tensor(WEIGHT).cuda() if self.use_gpu else torch.tensor(WEIGHT)
        self.schedule_weight = SCHEDULE
        self.freeze_layers = FREEZE_LAYERS
        # The total number of samples that have been classified before training the current task
        self.n_samples_prev = 0
        self.regularization_params_prev = {}
        # The number of samples from the current task that is learned that have been classified until now
        self.n_samples_cur = 0
        self.regularization_params_cur = {}
        self.n_tasks = 0
        self.prev_params = {}
        self._init_regularization_params()

    def _init_regularization_params(self) -> None:
        '''
            Initialize the importance weight omega for each parameter of the nn
        '''
        if self.n_samples_prev == 0:
            for name, param in self.model.named_parameters():
                if name in self.freeze_layers:
                    continue
                self.regularization_params_prev[name] = torch.zeros_like(param)
        # return if the current parameters have already been set to zero
        for name, param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            self.regularization_params_cur[name] = torch.zeros_like(param)

    def _before_train(self) -> None:
        self._save_prev_params()

    def _after_train(self,train_set:Dataset=None) -> None:
        self._update_weight()
        self._update_regularization_params()

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

    def _update_reg_params(self, outputs: torch.Tensor, batch_size: int) -> None:
        '''
            Updates the importance weight omega of each parameter. 
            Should be called whenever the model is evaluated
        '''
        output_l2 = nn.MSELoss(reduction='sum')
        targets = output_l2(outputs,torch.zeros_like(outputs))
        targets.backward(retain_graph=True)
        for name, param in self.model.named_parameters():
            if name not in self.regularization_params_prev:
                continue
            self.regularization_params_cur[name] = self._update_weights(self.regularization_params_cur[name],param,batch_size)

    def _update_weights(self,cur_param: torch.Tensor,p:torch.Tensor, batch_size:int):
        '''
            Updates the importance weight omega for a given parameter.
        '''
        prev_size = self.n_samples_cur
        self.n_samples_cur += batch_size
        cur_param.mul_(prev_size)
        gradient = p.grad.data.clone()
        cur_param.add_(gradient.abs())
        cur_param.div_(self.n_samples_cur)
        return cur_param

    def _compute_regularization_loss(self) -> torch.Tensor:
        '''
            Computes the regularization loss for one batch
        '''
        reg_loss = torch.tensor(0.0).cuda() if self.use_gpu else torch.tensor(0.0)
        for name,param in self.model.named_parameters():
            if name in self.freeze_layers:
                continue
            diff = (param - self.prev_params[name])**2 
            diff.mul_(self.regularization_params_prev[name])
            reg_loss += diff.sum()
        return self.weight * reg_loss

    def _after_pred_val(self,outputs:torch.Tensor,batch_size:int) -> None:
        self._update_reg_params(outputs,batch_size)

    def _update_weight(self) -> None:
        if not self.schedule_weight:
            return
        self.n_tasks += 1
        if self.n_tasks % 5 == 0:
            self.n_tasks = 0
            prev_weight = self.weight.item()
            self.weight = torch.tensor(2*prev_weight).cuda() if self.use_gpu else torch.tensor(2*prev_weight)

    def get_state(self) -> dict:
        return {
            'weight' : self.weight,
            'n_tasks' : self.n_tasks,
            'schedule_weight' : self.schedule_weight,
            'n_samples_prev' : self.n_samples_prev,
            'n_samples_cur' : self.n_samples_cur,
            'regularization_params_prev' : self.regularization_params_prev,
            'regularization_params_cur' : self.regularization_params_cur,
            'prev_params' : self.prev_params,
            'freeze_layers' : self.freeze_layers
        }
    
    def set_state(self,state:dict) -> None:
        self.weight = state['weight']
        self.n_tasks = state['n_tasks']
        self.schedule_weight = state['schedule_weight']
        self.n_samples_prev = state['n_samples_prev']
        self.n_samples_cur = state['n_samples_cur']
        self.regularization_params_prev = state['regularization_params_prev']
        self.regularization_params_cur = state['regularization_params_cur']
        self.prev_params = state['prev_params']
        self.freeze_layers = state['freeze_layers']