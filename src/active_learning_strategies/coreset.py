import random
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler
from .kCenterGreedy import kCenterGreedy

class CoreSet(Strategy):
    def __init__(self, model: nn.Module, data_unlabeled: Dataset, NO_CLASSES: int, test_loader: DataLoader, cfgs, device):
        super(CoreSet, self).__init__(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)

    def query(self):
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                    sampler=SubsetSequentialSampler(self.subset+self.labeled_set), 
                                    pin_memory=True)

        arg = self.get_kcg(unlabeled_loader)
        return arg

    def get_kcg(self, unlabeled_loader: DataLoader):
        labeled_data_size = self.BUDGET*self.cycle+self.INIT_BUDGET
        self.model['backbone'].eval()
        with torch.cuda.device(self.device):
            features = torch.tensor([]).cuda()

        with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(self.device):
                    inputs = inputs.cuda()
                _, features_batch, _ = self.model['backbone'](inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features.detach().cpu().numpy()
            new_av_idx = np.arange(len(self.subset),(len(self.subset) + labeled_data_size))
            sampling = kCenterGreedy(feat)  
            batch = sampling.select_batch_(new_av_idx, self.BUDGET)
            other_idx = [x for x in range(len(self.subset)) if x not in batch]
        return  other_idx + batch