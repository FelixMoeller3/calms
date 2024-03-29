import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import torch
from torch.nn import functional as F
import torch.nn as nn
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler

class BALD(Strategy):
    '''
        Implements the strategy Bayesian Active Learning by Disagreement (BALD) as proposed
        in the following paper: https://arxiv.org/pdf/1112.5745.pdf
    '''
    def __init__(self, model: nn.Module, data_unlabeled: Dataset, NO_CLASSES: int,BATCH:int,BUDGET:int,
        INIT_BUDGET:int, DROPOUT_ITER:int,LOOKBACK:int,USE_GPU:bool=False,**kwargs):
        super(BALD, self).__init__(model, data_unlabeled,NO_CLASSES,BATCH,BUDGET,INIT_BUDGET,LOOKBACK,USE_GPU)
        self.dropout_iter = DROPOUT_ITER

    def query(self) -> np.ndarray:
        if len(self.subset) <= self.BUDGET:
            arg = np.array([i for i in range(len(self.subset))])
        else:
            if not self.contains_dropout():
                print("Setting dropout iterations to 1 as the model does not contain dropout layers")
                dropout_iter = 1
            else:
                self.activate_dropout()
                dropout_iter = self.dropout_iter
            unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                        sampler=SubsetSequentialSampler(self.subset), 
                                        pin_memory=True)
            n_uPts = len(self.subset)
        
            # Heuristic:m G_X - F_X
            score_ALL = np.zeros(shape=(n_uPts, self.NO_CLASSES))
            all_entropy_dropout = np.zeros(shape=(n_uPts))

            for d in tqdm(
                range(dropout_iter),
                desc = "Dropout Iterations",
            ):
                probs = self.get_predict_prob(unlabeled_loader).cpu().numpy()
                score_ALL += probs

                # computing F_X
                dropout_score_log = np.log2(
                    probs + 1e-6
                )# add 1e-6 to avoid log(0)
                Entropy_Compute = -np.multiply(probs, dropout_score_log)
                Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

                all_entropy_dropout += Entropy_per_Dropout

            Avg_Pi= np.divide(score_ALL, self.dropout_iter)
            Log_Avg_Pi = np.log2(Avg_Pi + 1e-6)
            Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
            G_X = np.sum(Entropy_Avg_Pi, axis=1)
            F_X =np.divide(all_entropy_dropout, self.dropout_iter)
            U_X = G_X - F_X
            arg = np.argsort(-U_X)
        self.add_query(arg[:self.BUDGET])
        self.model.eval()
        return np.concatenate(self.previous_queries)

    def get_predict_prob(self, unlabeled_loader: DataLoader) -> torch.Tensor:
        #TODO: let this run on cuda when running on cluster
        #with torch.cuda.device(self.device):
        #    predic_probs = torch.tensor([]).cuda()
        predic_probs = torch.tensor([])
        if self.use_gpu:
            predic_probs = predic_probs.cuda()
        
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                #TODO: let this run on cuda when running on cluster
                #with torch.cuda.device(self.device):
                #    inputs = inputs.cuda()
                if self.use_gpu:
                    inputs = inputs.cuda()
                outputs = self.model(inputs)
                prob = F.softmax(outputs, dim=1)
                predic_probs = torch.cat((predic_probs, prob), 0)
        return predic_probs

    def contains_dropout(self) -> bool:
        '''
            Checks if the model contains dropout layers
        '''
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                return True
        return False

    def activate_dropout(self):
        '''
            Activates the dropout layers in the model 
        '''
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()