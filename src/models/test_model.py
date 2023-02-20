import torch.nn as nn
from functools import reduce
from torch import flatten
import torch
import torch.nn.functional as F

class testNN(nn.Module):
    def __init__(self, input_size, hidden_num, hidden_size, input_dropout, hidden_dropout, num_classes):
        super(testNN,self).__init__()
        '''
        self.input_size = input_size
        self.output_size = num_classes
        self.hidden_num = hidden_num
        self.hidden_size = hidden_size
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        
        self.layers = nn.ModuleList([
            # Input Layer
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout),

            # Hidden layers
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout)) * self.hidden_num),

            # Output layer
            nn.Linear(self.hidden_size, self.output_size)
        ])
        '''
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size,400)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(400,400)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(400,num_classes)
        

    def forward(self, x):
        out = x.view(-1,self.input_size)
        out = self.fc1(out)
        out = self.r1(out)
        out = self.fc2(out)
        out = self.r2(out)
        out = self.fc3(out)
        return out
        #return reduce(lambda x, l: l(x), self.layers,x.view(-1,self.input_size))

    def forward_embedding(self, x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        out = x.view(-1,self.input_size)
        out = self.fc1(out)
        out = self.r1(out)
        out = self.fc2(out)
        out = self.r2(out)
        emb = out.view(out.size(0),-1)
        return self.fc3(out),emb

    def get_embedding_dim(self) -> int:
        return self.fc3.in_features

class testConv(nn.Module):
    def __init__(self, input_dim:tuple[int], classes:int):
        super(testConv,self).__init__()
        after_conv_x = (input_dim[1] - 4)//2
        after_conv_x = (after_conv_x - 4)//2
        after_conv_y = (input_dim[2] - 4)//2
        after_conv_y = (after_conv_y - 4)//2
        self.layers_before = nn.ModuleList([
            nn.Conv2d(in_channels=input_dim[0],out_channels=20,kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(in_channels=20,out_channels=50,kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        ])
        self.layers_after = nn.ModuleList([
            nn.Linear(in_features=50*after_conv_x*after_conv_y,out_features=500),
            nn.ReLU()
        ])
        self.final = nn.Linear(in_features=500,out_features=classes)

    def forward(self, x:torch.Tensor):
        before_flatten = reduce(lambda x,l: l(x),self.layers_before,x)
        flattened = flatten(before_flatten,1)
        before_final = reduce(lambda x,l: l(x),self.layers_after,flattened)
        return self.final(before_final)

    def forward_embedding(self, x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        before_flatten = reduce(lambda x,l: l(x),self.layers_before,x)
        flattened = flatten(before_flatten,1)
        before_final = reduce(lambda x,l: l(x),self.layers_after,flattened)
        emb = before_final.view(before_final.size(0),-1)
        return self.final(before_final),emb

    def get_embedding_dim(self) -> int:
        return self.final.in_features

