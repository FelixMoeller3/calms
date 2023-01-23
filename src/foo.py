from models.test_model import testNN
import continual_learning_strategies as cl_strat
from continual_learning_strategies.ewc import ElasticWeightConsolidation
from torch.optim import SGD
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset

model = testNN(28*28,400,400,10)
optim = SGD(model.parameters(),lr=0.001)
mas_optim = cl_strat.mas.Weight_Regularized_SGD(model.parameters(),0.001,momentum=0.9)
mas_strat = cl_strat.mas.MAS(model,optim,10.0,nn.CrossEntropyLoss(),[])
#ewc_strat = cl_strat.ElasticWeightConsolidation(model,optim,nn.CrossEntropyLoss(),10.0)
mnist_train = datasets.MNIST('./data', train=True,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
mnist_test = datasets.MNIST('./data', train=False,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
indices_first_train = []
indices_second_train = []

for i in range(len(mnist_train)):
    if mnist_train.targets[i] in [0,2,4,6,8]:
        indices_first_train.append(i)
    else:
        indices_second_train.append(i)

mnist_train_first = Subset(mnist_train,indices_first_train)
mnist_train_second = Subset(mnist_train,indices_second_train)

indices_first_test = []
indices_second_test = []
for i in range(len(mnist_test)):
    if mnist_test.targets[i] in [0,2,4,6,8]:
        indices_first_test.append(i)
    else:
        indices_second_test.append(i)

mnist_test_first = Subset(mnist_test,indices_first_test)
mnist_test_second = Subset(mnist_test,indices_second_test)
 

mnist_train_first_loader = DataLoader(mnist_train_first, batch_size = 100, shuffle=True)
mnist_test_first_loader = DataLoader(mnist_test_first, batch_size = 100, shuffle=False)
mnist_first_loaders = {'train': mnist_train_first_loader, 'val': mnist_test_first_loader}
mas_strat.train(mnist_first_loaders,5)
#ewc_strat.train(mnist_first_loaders,mnist_first_dset_sizes,20)
print('First five digits of MNIST done. Starting with second five digits of MNIST')
'''
fashion_mnist_train = datasets.FashionMNIST('./data', train=True,download=True, transform=transforms.ToTensor())
fashion_mnist_test = datasets.FashionMNIST('./data', train=False,download=True, transform=transforms.ToTensor())
fashion_mnist_train_loader = DataLoader(fashion_mnist_train, batch_size = 100, shuffle=True)
fashion_mnist_test_loader = DataLoader(fashion_mnist_test, batch_size = 100, shuffle=False)
fashion_mnist_loaders = {'train': fashion_mnist_train_loader, 'val': fashion_mnist_test_loader}
fashion_mnist_dset_sizes = {'train': len(fashion_mnist_train), 'val': len(fashion_mnist_test)}
'''
mnist_train_second_loader = DataLoader(mnist_train_second, batch_size = 100, shuffle=True)
mnist_test_second_loader = DataLoader(mnist_test_second, batch_size = 100, shuffle=False)
mnist_second_loaders = {'train': mnist_train_second_loader, 'val': mnist_test_second_loader}
mnist_second_dset_sizes = {'train': len(mnist_train_second), 'val': len(mnist_test_second)}

mas_strat.train(mnist_second_loaders,5)
#ewc_strat.train(mnist_second_loaders, mnist_second_dset_sizes,20)
print('Testing first digits of MNIST after training all digits')
mas_strat._run_val_epoch(mnist_test_first_loader)
#ewc_strat._run_val_epoch(mnist_test_first_loader,len(mnist_test_first))
print('Testing second digits of MNIST after training all digits')
mas_strat._run_val_epoch(mnist_test_second_loader)
#ewc_strat._run_val_epoch(mnist_test_second_loader,len(mnist_test_second))

# Print gradient and parameters of network in each epoch