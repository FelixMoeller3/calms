from torch.utils.data import Dataset
import torch

class Softmax_label_set(Dataset):
    '''
        This dataset class is used to store datasets where the labels are an array of values.
        In this project the class is used to store the softmax output of the target model 
    '''
    def __init__(self, subset: Dataset, num_classes:int):
        self.subset = subset
        self.targets = torch.zeros(len(subset),num_classes)

        
    def __getitem__(self, index):
        x, _ = self.subset[index]
        return x, self.targets[index]
        
    def __len__(self):
        return len(self.subset)
