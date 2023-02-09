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
        is queried next.
    '''
    def __init__(self, model: nn.Module, data_unlabeled: Dataset, NO_CLASSES:int, test_loader: DataLoader, cfgs, device):
        super(LC, self).__init__(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)

    def query(self) -> np.ndarray:
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                    sampler=SubsetSequentialSampler(self.subset), 
                                    pin_memory=True)
        probs = self.get_predict_prob(unlabeled_loader)
        probs_max = np.amax(probs,1)
        arg = np.argsort(-probs_max)
        return arg

    def get_predict_prob(self, unlabeled_loader: DataLoader) -> torch.Tensor:
        self.model['backbone'].eval()
        with torch.cuda.device(self.device):
            predic_probs = torch.tensor([]).cuda()

        with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(self.device):
                    inputs = inputs.cuda()
                predict, _, _ = self.model['backbone'](inputs)
                prob = F.softmax(predict, dim=1)
                predic_probs = torch.cat((predic_probs, prob), 0)

        return predic_probs