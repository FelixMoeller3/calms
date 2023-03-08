import torch.nn as nn
from functools import reduce
from torch import flatten
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(self.pool(out))
        return out
    

class ThiefConvNet(nn.Module):

    def __init__(self,num_conv_blocks=4,input_channels=3,num_classes=10,input_dim:int=64):
        super(ThiefConvNet,self).__init__()
        blocks = []
        for i in range(num_conv_blocks):
            in_channels = input_channels if i == 0 else 16*(2**i)
            out_channels = 32*(2**i)
            blocks.append(ConvBlock(in_channels,out_channels))
            input_dim //= 2
        self.before_final = nn.Sequential(*blocks)
        self.final = nn.Linear(out_channels*input_dim*input_dim,num_classes)

    def forward(self, x):
        before_flatten = self.before_final(x)
        flattened = before_flatten.view(before_flatten.size(0),-1)
        return self.final(flattened)
    

    def forward_embedding(self, x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        before_flatten = self.before_final(x)
        flattened = before_flatten.view(before_flatten.size(0),-1)
        return self.final(flattened),flattened
    
    def get_embedding_dim(self) -> int:
        return self.final.in_features
