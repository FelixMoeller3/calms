import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler

class Entropy(Strategy):
    def __init__(self, model: nn.Module, data_unlabeled, NO_CLASSES: int, test_loader: DataLoader, 
        batch:int,budget:int, init_budget:int, device):
        super(Entropy, self).__init__(model, data_unlabeled, NO_CLASSES, test_loader,batch,budget,init_budget,device)

    def query(self):
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                    sampler=SubsetSequentialSampler(self.subset), 
                                    pin_memory=True)
        probs = self.get_predict_prob(unlabeled_loader)
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(1).cpu()
        arg = np.argsort(U)
        return arg

    def get_predict_prob(self, unlabeled_loader: DataLoader):
        self.model.eval()
        with torch.cuda.device(self.device):
            predic_probs = torch.tensor([]).cuda()

        with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(self.device):
                    inputs = inputs.cuda()
                predict, _, _ = self.model(inputs)
                prob = F.softmax(predict, dim=1)
                predic_probs = torch.cat((predic_probs, prob), 0)

        return predic_probs