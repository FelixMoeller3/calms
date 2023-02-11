import torch.nn as nn
from functools import reduce
from torch import flatten

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

    def get_embedding_dim(self) -> int:
        return self.fc3.in_features

class testConv(nn.Module):
    def __init__(self, numChannels:int, classes:int):
        super(testConv,self).__init__()
        self.layers_before = nn.ModuleList([
            nn.Conv2d(in_channels=numChannels,out_channels=20,kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(in_channels=20,out_channels=50,kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        ])
        self.layers_after = nn.ModuleList([
            nn.Linear(in_features=800,out_features=500),
            nn.ReLU(),

            nn.Linear(in_features=500,out_features=classes),
            nn.LogSoftmax(dim=1)
        ])

    def forward(self, x):
        before_flatten = reduce(lambda x,l: l(x),self.layers_before,x)
        flattened = flatten(before_flatten,1)
        return reduce(lambda x,l: l(x),self.layers_after,flattened)