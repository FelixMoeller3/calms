from typing import List
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
# Custom
# from influence_function import calc_influence_function
# from Influence_function import calc_influence_function, calc_influence_function_param

class Strategy:
    '''
        Base class for continual learning strategies
    '''
    def __init__(self, model:nn.Module, data_unlabeled, NO_CLASSES:int,
     batch:int,budget:int, init_budget:int, look_back:int, use_gpu:bool):
        self.model = model
        self.data_unlabeled = data_unlabeled
        self.subset = []
        self.labeled_set = []
        self.cycle = 0
        self.NO_CLASSES = NO_CLASSES
        self.use_gpu = use_gpu
        self.init_set_seed = 0
        self.look_back = look_back
        self.previous_queries = []
        self.num_saved_queries = 0
        self.BATCH = batch
        self.BUDGET = budget
        self.INIT_BUDGET = init_budget

    def add_query(self,new_query:np.ndarray) -> np.ndarray:
        self.num_saved_queries += 1
        if self.num_saved_queries > self.look_back:
            self.previous_queries.pop(0)
        self.previous_queries.append(new_query)


    def query(self):
        pass

    def seed_random_init_set(self, init_set_seed):
        self.init_set_seed = init_set_seed
    
    def feed_current_state(self, cycle:int, subset:List[int], labeled_set: List[int]):
        self.subset = subset
        self.labeled_set = labeled_set
        self.cycle = cycle

