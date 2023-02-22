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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, train_deterministic=False, fixed_model_parameter_seed=None):
        super(ResNet, self).__init__()
        if fixed_model_parameter_seed:
            torch.manual_seed(fixed_model_parameter_seed)

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.adaptive_avg_pool2d(out4, 1)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out

    def forward_embedding(self, x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.adaptive_avg_pool2d(out4, 1)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out, outf

    def get_embedding_dim(self) :
        return self.linear.in_features

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out