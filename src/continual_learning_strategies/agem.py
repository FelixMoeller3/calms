import torch.nn as nn
from .cl_base import ContinualLearningStrategy
import torch
import torch.optim.lr_scheduler as lr_scheduler
import random
from torch.utils.data import Dataset

class AGem(ContinualLearningStrategy):

    def __init__(self,model:nn.Module,optim: torch.optim.Optimizer, scheduler: lr_scheduler._LRScheduler,crit: nn.CrossEntropyLoss,
    PATTERNS_PER_EXPERIENCE:int=30,SAMPLE_SIZE:int=100,USE_GPU:bool=False,clip_grad: float=2.0,state_dict:dict=None,**kwargs):
        super(AGem,self).__init__(model,optim,scheduler,crit,USE_GPU,clip_grad)
        if state_dict is not None:
            self.set_state(state_dict)
            return
        self.patterns_per_experience = PATTERNS_PER_EXPERIENCE
        self.sample_size = SAMPLE_SIZE
        self.buffer_data = torch.empty(0)
        self.buffer_targets = torch.empty(0)
        self.reference_gradients = None

    def _before_training_iteration(self):
        """
        Compute reference gradient on memory sample.
        """
        if len(self.buffer_data) == 0:
            return

        self.model.train()
        self.optim.zero_grad()
        mb = self._sample_from_memory()
        data, labels = mb
        if self.use_gpu:
            data, labels = data.cuda(), labels.cuda()

        out = self.model(data)
        loss = self.crit(out, labels)
        loss.backward()
        # gradient can be None for some head on multi-headed models
        self.reference_gradients = [
            p.grad.view(-1)
            if p.grad is not None
            else torch.zeros(p.numel()).cuda() if self.use_gpu else torch.zeros(p.numel())
            for _, p in self.model.named_parameters()
        ]
        self.reference_gradients = torch.cat(self.reference_gradients)
        self.optim.zero_grad()

    def _sample_from_memory(self) -> None:
        num_to_sample = min(len(self.buffer_data),self.sample_size)
        data = torch.empty([num_to_sample]+ list(self.buffer_data[0].shape))
        if isinstance(self.buffer_targets[0],torch.Tensor):
            targets = torch.zeros([num_to_sample] + list(self.buffer_targets[0].shape))
        else:
            targets = torch.zeros(num_to_sample,dtype=torch.long)
        indexes = random.sample([i for i in range(len(self.buffer_data))],num_to_sample)
        for i,index in enumerate(indexes):
            data[i] = self.buffer_data[index]
            targets[i] = self.buffer_targets[index]
        return (data,targets)

    @torch.no_grad()
    def after_backward(self):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffer_data) == 0:
            return
        
        current_gradients = [
            p.grad.view(-1)
            if p.grad is not None
            else torch.zeros(p.numel()).cuda() if self.use_gpu else torch.zeros(p.numel())
            for _, p in self.model.named_parameters()
        ]
        current_gradients = torch.cat(current_gradients)

        assert (
            current_gradients.shape == self.reference_gradients.shape
        ), "Different model parameters in AGEM projection"

        dotg = torch.dot(current_gradients, self.reference_gradients)
        if dotg < 0:
            alpha2 = dotg / torch.dot(
                self.reference_gradients, self.reference_gradients
            )
            grad_proj = (
                current_gradients - self.reference_gradients * alpha2
            )

            count = 0
            for _, p in self.model.named_parameters():
                n_param = p.numel()
                if p.grad is not None:
                    p.grad.copy_(
                        grad_proj[count : count + n_param].view_as(p)
                    )
                count += n_param


    def _after_train(self,train_set: Dataset=None) -> None:
        num_elems = min(len(train_set),self.patterns_per_experience)
        indexes = random.sample([i for i in range(len(train_set))],num_elems)
        data = torch.zeros([num_elems] + list(train_set[0][0].shape))
        if isinstance(train_set[0][1],torch.Tensor):
            targets = torch.zeros([num_elems] + list(train_set[0][1].shape))
        else:
            targets = torch.zeros(num_elems,dtype=torch.long)
        for i,index in enumerate(indexes):
            data[i],targets[i] = train_set[index]
        
        if len(self.buffer_data) == 0:
            self.buffer_data = data
            self.buffer_targets = targets
        else:
            self.buffer_data = torch.cat([self.buffer_data,data])
            self.buffer_targets = torch.cat([self.buffer_targets,targets])

    def get_state(self) -> dict:
        return {
            'patterns_per_experience': self.patterns_per_experience,
            'sample_size': self.sample_size,
            'buffer_data': self.buffer_data,
            'buffer_targets': self.buffer_targets,
            'reference_gradients': self.reference_gradients
        }
        
    def set_state(self,state:dict) -> None:
        self.patterns_per_experience = state['patterns_per_experience']
        self.sample_size = state['sample_size']
        self.buffer_data = state['buffer_data']
        self.buffer_targets = state['buffer_targets']
        self.reference_gradients = state['reference_gradients']
