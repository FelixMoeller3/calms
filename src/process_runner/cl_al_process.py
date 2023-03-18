from .process_base import BaseProcess
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from torch.utils.data import DataLoader,Dataset
from typing import Callable,List


class ClAlProcess(BaseProcess):

    def __init__(self,activeLearningStrategy: Strategy,continualLearningStrategy: ContinualLearningStrategy,train_set: Dataset,
                 val_set: Dataset,batch_size:int,num_cycles:int,num_epochs:int,continual:int,optimizer_config:dict,
                 optimizer_builder: Callable,init_mode:str='random',state_dir:str=None):
        '''
            :param targetModel: the target model in the model stealing process (i.e. the one that will be stolen). Needs to be pretrained!!
            :param activeLearningStrategy: the active learning strategy to use in the model stealing process.
            :param continualLearningStrategy: the continual learning strategy to use in the model stealing process. Note that this class already contains the
            substitute model.
        '''
        super(ClAlProcess,self).__init__(activeLearningStrategy,continualLearningStrategy,train_set,val_set,batch_size,num_cycles,
                                         num_epochs,continual,optimizer_builder,optimizer_config,init_mode,False,state_dir)


    def continual_learning(self,start_cycle:int=0,labeled_set:List[int]=[],unlabeled_set:List[int]=[],score_list:List[float]=[]) -> tuple[List[float],List[List[float]]]:
        '''
            Runs a combined continual and active learning approach where instead of querying the target model the actual label is used.
            :param train_set: the dataset that the model will be trained on.
            :param val_set: the dataset used to validate the performance of the model
            :param batch_size: the batch size used to train the model.
            :param num_cycles: The number of cycles used in the active learning process.
        '''
        val_loader = DataLoader(self.val_set,self.batch_size,shuffle=True)
        loaders_dict = {'train': None, 'val': val_loader}
        if start_cycle == 0:
            labeled_set,unlabeled_set = self._before_first_cycle(loaders_dict,score_list)
        
        for i in range(start_cycle,self.num_cycles):
            print(f'Running cycle {i+1}/{self.num_cycles}')
            labeled_set,unlabeled_set = self._query_cycle(i,labeled_set,unlabeled_set,loaders_dict,score_list)
            if self.state_dir is not None:
                state_dict = {'start_cycle': i+1, 'labeled_set': labeled_set, 'unlabeled_set': unlabeled_set,'score_list': score_list}
                self._save_state(state_dict)
        return score_list   
