import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler
from .kCenterGreedy import kCenterGreedy

class CoreSet(Strategy):
    '''
        Implements the strategy CoreSet as proposed
        in the following paper: https://arxiv.org/pdf/1708.00489.pdf),
    '''
    def __init__(self, model: nn.Module, data_unlabeled: Dataset, NO_CLASSES: int,
        BATCH:int,BUDGET:int, INIT_BUDGET:int, LOOKBACK:int, USE_GPU:bool=False,**kwargs):
        super(CoreSet, self).__init__(model, data_unlabeled, NO_CLASSES,BATCH,BUDGET,INIT_BUDGET,LOOKBACK,USE_GPU)

    def query(self):
        if len(self.subset) <= self.BUDGET:
            arg = np.array([i for i in range(len(self.subset))])
        else:
            unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                        sampler=SubsetSequentialSampler(self.subset+self.labeled_set), 
                                        pin_memory=True)

            arg = self.get_kcg(unlabeled_loader)
        self.add_query(arg[:self.BUDGET])
        return np.concatenate(self.previous_queries)

    def get_kcg(self, unlabeled_loader: DataLoader):
        labeled_data_size = self.BUDGET*self.cycle+self.INIT_BUDGET
        self.model.eval()
        #TODO: let this run on cuda when running on cluster
        #with torch.cuda.device(self.device):
        #    features = torch.tensor([]).cuda()
        features = torch.tensor([])
        if self.use_gpu:
            features = features.cuda()

        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                #TODO: let this run on cuda when running on cluster
                #with torch.cuda.device(self.device):
                #    inputs = inputs.cuda()
                if self.use_gpu:
                    inputs = inputs.cuda()
                _, features_batch = self.model.forward_embedding(inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features.detach().cpu().numpy()
            new_av_idx = np.arange(len(self.subset),(len(self.subset) + labeled_data_size))
            sampling = kCenterGreedy(feat)  
            batch = sampling.select_batch_(new_av_idx, self.BUDGET)
        return batch