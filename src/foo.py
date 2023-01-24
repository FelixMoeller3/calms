from models.test_model import testNN, testConv
import continual_learning_strategies as cl_strat
from continual_learning_strategies.ewc import ElasticWeightConsolidation
from torch.optim import SGD
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import MNIST
from avalanche.benchmarks import nc_benchmark,SplitMNIST
from avalanche.training import MAS,Replay,EWC
from avalanche.models import IncrementalClassifier,MultiHeadClassifier
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    bwt_metrics,
)
from avalanche.training.plugins import EvaluationPlugin

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
benchmark = SplitMNIST(5,shuffle=False,return_task_id=True,class_ids_from_zero_in_each_exp=True,
train_transform=transforms.Compose([lambda x: x.view(784)]),eval_transform=transforms.Compose([lambda x: x.view(784)]))

eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        #loggers=[interactive_logger, tensorboard_logger],
    )

#model = testNN(28*28,1,400,0.2,0.5,10)
#model = testConv(1,10)
model = MultiHeadClassifier(in_features=784)
optim = SGD(model.parameters(),lr=0.005)
#mas_optim = cl_strat.mas.Weight_Regularized_SGD(model.parameters(),0.001,momentum=0.9)
#mas_strat = cl_strat.mas.MAS(model,optim,10.0,nn.CrossEntropyLoss(),[])

#ewc_strat = cl_strat.ElasticWeightConsolidation(model,optim,nn.CrossEntropyLoss(),100)
strat = EWC(model,optim,nn.CrossEntropyLoss(),1.0,train_epochs=10)
#strat = MAS(model,optim,nn.CrossEntropyLoss(),train_epochs=15,train_mb_size=64)
#mas_strat = Replay(model,optim,nn.CrossEntropyLoss(),train_mb_size=100,train_epochs=10)
'''
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
ewc_strat.train(train_datasets,test_datasets,5)


for i in range(len(train_datasets)):
    ewc_strat.eval(test_datasets[i])
'''
for i in range(len(benchmark.train_stream)):
    cur_exp = benchmark.train_stream[i]
    print("Start training on experience ", cur_exp.current_experience)
    strat.train_mb_size = 100
    strat.eval_mb_size = 100
    strat.train(cur_exp)
    print("End training on experience", cur_exp.current_experience)
    print("Computing accuracy on the test set")
    strat.eval(benchmark.test_stream[:])