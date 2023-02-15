from typing import List
import torch.nn as nn
from .cl_base import ContinualLearningStrategy
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

class Alasso(ContinualLearningStrategy):
    '''
        Implementation of Asymmetric Loss Approximation by Single-side Overestimation (ALASSO) proposed in the following paper:
        https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Continual_Learning_by_Asymmetric_Loss_Approximation_With_Single-Side_Overestimation_ICCV_2019_paper.pdf
        The implementation is based on the authors' implementation which can be found at: https://github.com/dmpark04/alasso
    '''

    def __init__(self,model:nn.Module,optim: torch.optim.Optimizer,crit: nn.CrossEntropyLoss,
    weight:float=1.0,weight_prime:float=1.0,a:float=1.0,a_prime:float=1.0,epsilon:float=1e-4):
        '''
            :param model: The model that should be trained using continual learning.
            :param optim: The optimizer to be used during training. Beware: When using Alasso, one should set a rather low learning rate as Alasso easily overshoots when using a medium to high learning rate.
            :param crit: The criterion function to use. Usually, this is Cross-Entropy loss.
            :param weight: The weight that should be used to weigh the surrogate loss. See the paper for details (it is called c there).
            :param weight_prime: This corresponds to c' in the paper. See section 4.4 (Parameter decoupling) for details.
            :param a: The parameter a controls the overestimation of the loss. See equation (5) of the paper for details.
            :param a_prime: This corresponds to a' in the paper. See section 4.4 (Parameter decoupling) for details.
            :param epsilon: This parameter is added to the denominator in equation (7) to make sure we are not dividing by zero. It should always be > 0.
        '''
        #TODO: Issue warning when parameter a is <=1 
        super(Alasso,self).__init__(model,optim,crit)
        self.weight = weight
        self.weight_prime = weight_prime
        self.a = a
        self.a_prime = a_prime
        self.epsilon = epsilon
        # Dict to save previous weights
        self.weights = {}
        self._save_weights()
        # Dict used to compute values to compute upper term in fraction for omega
        self.grads2 = {}
        self._compute_grads2()
        # Dict used to save omega values
        self.omegas = {}
        self._compute_omegas()
        # Dict used to save the gradients
        self.unreg_grads = {}
        self._compute_unreg_grads()
        # Dict used to save the weight deltas between two optimizer steps
        self.deltas = {}
        self._compute_deltas(compute_diff=False)
        
    
    def _save_weights(self) -> None:
        '''
            Saves the models current weights. Usually called after each optimizer step.
        '''
        for name,param in self.model.named_parameters():
            self.weights[name] = param.detach().clone()

    def _compute_grads2(self) -> None:
        '''
            Compute the values for the upper term in fraction for omega calculation. Usually called
            after each optimizer step.
        '''
        if not self.grads2:
            for name,param in self.model.named_parameters():
                self.grads2[name] = torch.zeros_like(param)
            return
        for name, param in self.model.named_parameters():
            self.grads2[name] = self.grads2[name] - self.unreg_grads[name] * self.deltas[name]


    def _compute_omegas(self) -> None:
        '''
            Compute the Omega values for each parameter. Usually calculated after each task.
        '''
        if not self.omegas:
            for name,param in self.model.named_parameters():
                self.omegas[name] = torch.zeros_like(param)
            return
        for name, param in self.model.named_parameters():
            self.omegas[name] = torch.where(
                param < self.weights[name],
                torch.fill(torch.zeros_like(self.omegas[name]),-1.0),
                torch.ones_like(self.omegas[name])
            ) * (self.grads2[name] - self.weight_prime * self.asymmetric_loss_func(name,param,self.a_prime))/((self.weights[name]-param)*(self.weights[name]-param) + self.epsilon)

    def asymmetric_loss_func(self,name:str,param:torch.Tensor,a:float) -> torch.Tensor:
        '''
            Computes the asymmetric loss
            :param name: the name of the current parameter of the model. Used to access the variable dicts.
            :param param: the values of the current parameter.
            :param a: the a value to use. See section 'parameter decoupling' in the paper for details
            :param epsilon: the epsilon value to use. Used so that the loss is overapproximated even if omega is 0
        '''
        return torch.where(
                self.omegas[name] > 0.0,
                torch.where(
                    param < self.weights[name],
                    self.omegas[name] * ((param - self.weights[name])*(param - self.weights[name])),
                    (self.omegas[name] * a + self.epsilon) * ((param - self.weights[name])*(param - self.weights[name]))
                ),
                torch.where(
                    param < self.weights[name],
                    (-1.0 * self.omegas[name] * a + self.epsilon) * ((param - self.weights[name])*(param - self.weights[name])),
                    -1.0 * self.omegas[name] * ((param - self.weights[name])*(param - self.weights[name]))
                )
            )

    def _compute_unreg_grads(self) -> None:
        '''
            Computes the gradient of the unregularized loss for each parameter
        '''
        if not self.unreg_grads:
            for name,param in self.model.named_parameters():
                self.unreg_grads[name] = torch.zeros_like(param)
            return
        for name,param in self.model.named_parameters():
            self.unreg_grads[name] = param.grad.detach().clone()

    def _compute_deltas(self,compute_diff:bool=True):
        '''
            Helper function used to compute the deltas between the weights in each step
            :param compute_diff: If set to True, compute the difference between current and
            previous parameters. Otherwise, just save the current model parameters
        '''
        if not compute_diff:
            for name,param in self.model.named_parameters():
                self.deltas[name] = param.detach().clone()
            return
        for name, param in self.model.named_parameters():
            self.deltas[name] = param - self.deltas[name]


    def train(self, dataloaders: dict[str,DataLoader], num_epochs: int,result_list:List[float]=[]) -> None:
        '''
            Trains the model for num_epoch epochs using the dataloaders 'train' and 'val' in the dataloaders dict
        '''
        start_time = time.time()
        self._compute_omegas()
        self._save_weights()
        self.grads2 = {}
        self._compute_grads2()
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            self._run_train_epoch(dataloaders['train'],last_epoch=epoch==num_epochs-1)
            log_list = None if epoch < num_epochs-1 else result_list
            self._run_val_epoch(dataloaders['val'],log_list)
        # Update Omegas
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def _run_train_epoch(self,dataloader: DataLoader,last_epoch:bool=False) -> None:
        '''
            Runs one epoch of the training procedure with the data given by the dataloader.
        '''
        self.model.train(True)
        total_loss = 0.0
        correct_predictions = 0
        for data in tqdm(dataloader):
            self.optimizer.zero_grad()
            self._compute_deltas(compute_diff=False)
            inputs, labels = data
            outputs = self.model(inputs)
            unreg_loss = self.crit(outputs, labels)
            unreg_loss.backward()
            self._compute_unreg_grads()
            reg_loss = self._compute_consolidation_loss()
            #retain_graph = (not last_epoch) or i<num_batches-1
            #start = time.time()
            reg_loss.backward(retain_graph=True)
            self.optimizer.step()
            self._compute_deltas()
            self._compute_grads2()
            _, preds = torch.max(outputs.data, 1)
            total_loss += reg_loss.item() + unreg_loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def calc_num_node_in_grad_fn(self,grad_fn):
        result = 0
        if grad_fn is not None:
            result += 1
            if hasattr(grad_fn, 'next_functions'):
                for f in grad_fn.next_functions:
                    result += self.calc_num_node_in_grad_fn(f)
        return result
    
    def _run_val_epoch(self,dataloader: DataLoader,log_list:List[float]=None) -> None:
        '''
            Runs one validation epoch using the dataloader which contains the validation data. 
        '''
        total_loss = 0.0
        correct_predictions = 0
        self.model.train(False)
        for data in tqdm(dataloader):
            inputs, labels = data

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data,1)
            loss = self.crit(outputs,labels)
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)
        if log_list is not None:
            log_list.append(epoch_acc)
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def _compute_consolidation_loss(self) -> torch.Tensor:
        '''
            Computes the loss added to the standard loss via the regularization term.
        '''
        loss = torch.tensor(0.0)
        for name,param in self.model.named_parameters():
            loss += self.asymmetric_loss_func(name,param,self.a).sum()
        return loss * self.weight

