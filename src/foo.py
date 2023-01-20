from models.test_model import testNN
import continual_learning_strategies as cl_strat
from continual_learning_strategies.ewc import ElasticWeightConsolidation
from torch.optim import SGD
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset

model = testNN(28*28,100,50,10)
optim = SGD(model.parameters(),lr=0.1)
mas_strat = cl_strat.mas.MAS(model,optim,0.001,nn.CrossEntropyLoss(),[])
mnist_train = datasets.MNIST('./data', train=True,download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST('./data', train=False,download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_train, batch_size = 100, shuffle=False)
loaders = {'train': train_loader, 'val': test_loader}
dset_sizes = {'train': len(mnist_train), 'val': len(mnist_test)}
mas_strat.train(loaders, dset_sizes,3)

#for key in mas_strat.omegas:
#    print(mas_strat.omegas[key])
