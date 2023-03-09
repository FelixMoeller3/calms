import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler

class LC(Strategy):
    '''
        Implements the 'least confident' strategy where the data that the model is least confident predicting
        is queried next, first proposed in the following paper: https://dl.acm.org/doi/pdf/10.1145/219587.219592
    '''
    def __init__(self, model: nn.Module, data_unlabeled: Dataset, NO_CLASSES:int,BATCH:int,
        BUDGET:int, INIT_BUDGET:int, LOOKBACK:int, USE_GPU=False,**kwargs):
        super(LC, self).__init__(model, data_unlabeled, NO_CLASSES,BATCH,BUDGET,INIT_BUDGET,LOOKBACK,USE_GPU)

    def query(self) -> np.ndarray:
        if len(self.subset) <= self.BUDGET:
            arg = np.array([i for i in range(len(self.subset))])
        else:
            unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                        sampler=SubsetSequentialSampler(self.subset), 
                                        pin_memory=True)
            probs = self.get_predict_prob(unlabeled_loader)
            arg = np.argsort(probs)
        self.add_query(arg[:self.BUDGET])
        return np.concatenate(self.previous_queries)

    def get_predict_prob(self, unlabeled_loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        #TODO: enable when running on cluster
        #with torch.cuda.device(self.device):
        #    predic_probs = torch.tensor([]).cuda()
        predic_probs = torch.tensor([])
        if self.use_gpu:
            predic_probs = predic_probs.cuda()
        with torch.no_grad():
            for inputs,_ in unlabeled_loader:
                #TODO: enable when running on cluster
                if self.use_gpu:
                    inputs = inputs.cuda()
                #with torch.cuda.device(self.device):
                #    inputs = inputs.cuda()
                predictions = self.model(inputs)
                prob = F.softmax(predictions, dim=1)
                predict_vals,_ = torch.max(prob, 1)
                predic_probs = torch.cat((predic_probs, predict_vals), 0)
        predic_probs = predic_probs.cpu()
        return predic_probs