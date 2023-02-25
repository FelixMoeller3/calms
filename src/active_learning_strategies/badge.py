import random
from typing import List
import torch
import numpy as np
from copy import deepcopy
from scipy import stats
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
import torch.nn as nn
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler

class Badge(Strategy):
    '''
        Implements the strategy Batch Active learning by Diverse Gradient Embeddings (BADGE) as proposed
        in the following paper: https://arxiv.org/pdf/1906.03671.pdf
    '''
    def __init__(self, model: nn.Module, data_unlabeled: Dataset, NO_CLASSES: int,
        BATCH:int,BUDGET:int, INIT_BUDGET:int, LOOKBACK: int, USE_GPU:bool=False,**kwargs):
        super(Badge, self).__init__(model, data_unlabeled, NO_CLASSES,BATCH,BUDGET,INIT_BUDGET, LOOKBACK, USE_GPU)

    def query(self) -> List[int]:
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                    sampler=SubsetSequentialSampler(self.subset), 
                                    pin_memory=True)

        gradEmbedding = self.get_grad_embedding(unlabeled_loader, len(self.subset)).numpy()
        print('features shape: {}'.format(gradEmbedding.shape))
        print(self.BUDGET)
        arg = self.init_centers(gradEmbedding)
        self.add_query(arg[:self.BUDGET])
        return np.concatenate(self.previous_queries)

    def get_grad_embedding(self, unlabeled_loader: DataLoader, len_ulb: int) -> torch.Tensor:
        embDim = self.model.get_embedding_dim()
        self.model.eval()
        nLab = self.NO_CLASSES
        embedding = np.zeros([len_ulb, embDim*nLab])
        ind = 0
        print('embedding shape {}'.format(embedding.shape))
        with torch.no_grad():
            for x, y in unlabeled_loader:
                # print(idxs)
                #TODO: let this run on cuda when running on cluster
                #with torch.cuda.device(self.device):
                if self.use_gpu:
                    x = x.cuda()
                scores, features_batch = self.model.forward_embedding(x)
                features_batch = features_batch.data.cpu().numpy()
                batchProbs = F.softmax(scores, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    # print(idxs[j],ind)
                    for c in range(nLab):
                        # if j==0:
                        #     print(c, idxs)
                        # print(idxs[j],ind)
                        if c == maxInds[j]:
                            embedding[ind][embDim * c : embDim * (c+1)] = deepcopy(features_batch[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[ind][embDim * c : embDim * (c+1)] = deepcopy(features_batch[j]) * (-1 * batchProbs[j][c])
                    ind += 1
            # print(ind)
            return torch.Tensor(embedding)

    def init_centers(self, X) -> List[int]:
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < self.BUDGET:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            # if sum(D2) == 0.0: pdb.set_trace()
            assert sum(D2) != 0.0
            D2 = D2.ravel().astype(float)
            
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        others = [i for i in range(len(X)) if i not in indsAll]
        return others + indsAll