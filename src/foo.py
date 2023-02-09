from models.test_model import testNN
import continual_learning_strategies as cl_strat
from torch.optim import SGD
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import MNIST

mnist_train = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
mnist_test = MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )



model = testNN(28*28,1,400,0.2,0.5,10)
#model = testConv(1,10)
optim = SGD(model.parameters(),lr=0.01)
#mas_optim = cl_strat.mas.Weight_Regularized_SGD(model.parameters(),0.001,momentum=0.9)
#mas_strat = cl_strat.MAS(model,optim,0.0,nn.CrossEntropyLoss(),[])
#mas_strat = cl_strat.ElasticWeightConsolidation(model,optim,nn.CrossEntropyLoss(),1.0)
#mas_strat = cl_strat.IMM(model,optim,nn.CrossEntropyLoss(),alphas=[0.6,0.4],weight=0.01,mean=False)
mas_strat = cl_strat.Alasso(model,optim,nn.CrossEntropyLoss(),0.8,0.7,1.5,0.75)
#mas_strat = cl_strat.Naive(model,optim,nn.CrossEntropyLoss())

#ewc_strat = cl_strat.ElasticWeightConsolidation(model,optim,nn.CrossEntropyLoss(),100)
#strat = EWC(model,optim,nn.CrossEntropyLoss(),1.0,train_epochs=10)
#strat = MAS(model,optim,nn.CrossEntropyLoss(),train_epochs=15,train_mb_size=64)
#mas_strat = Replay(model,optim,nn.CrossEntropyLoss(),train_mb_size=100,train_epochs=10)
mnist_train = datasets.MNIST('./data', train=True,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
mnist_test = datasets.MNIST('./data', train=False,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
mnist_train_first, mnist_train_second, mnist_train_third = random_split(mnist_train,(20000,20000,20000))
train_loader_one = DataLoader(mnist_train_first,100,shuffle=True)
train_loader_two = DataLoader(mnist_train_second,100,shuffle=True)
train_loader_three = DataLoader(mnist_train_third,100,shuffle=True)
test_loader = DataLoader(mnist_test,100,shuffle=True)
d1 = {'train': train_loader_one,'val': test_loader}
d2 = {'train': train_loader_two,'val': test_loader}
d3 = {'train': train_loader_three,'val': test_loader}
mas_strat.train(d1,5)
mas_strat.train(d2,5)
mas_strat.train(d3,5)

mas_strat._run_val_epoch(train_loader_one)
mas_strat._run_val_epoch(train_loader_two)
'''
indices_first_train = []
indices_second_train = []
all_digits = [[0,1],[2,3],[4,5],[6,7],[8,9]]
train_datasets = []
test_datasets = []
indices = [[],[],[],[],[]]

for i in range(len(mnist_train)):
    cur_target = mnist_train.targets[i]
    indices[cur_target//2].append(i)
for i in range(len(indices)):
    train_datasets.append(Subset(mnist_train,indices[i]))
indices = [[],[],[],[],[]]
for i in range(len(mnist_test)):
    cur_target = mnist_test.targets[i]
    indices[cur_target//2].append(i)
for i in range(len(indices)):
    test_datasets.append(Subset(mnist_test,indices[i]))
    

#for i in range(len(train_datasets)):
#    ewc_strat.train(train_datasets[i],test_datasets[i],5)
for i in range(len(benchmark.train_stream)):
    cur_exp = benchmark.train_stream[i]
    print("Start training on experience ", cur_exp.current_experience)
    strat.train_mb_size = 100
    strat.eval_mb_size = 100
    strat.train(cur_exp)
    print("End training on experience", cur_exp.current_experience)
    print("Computing accuracy on the test set")
    strat.eval(benchmark.test_stream[:])
'''