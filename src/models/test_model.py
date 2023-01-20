import torch.nn as nn

class testNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(testNN,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size,num_classes)

    def forward(self, x):
        out = x.view(x.shape[0],-1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out