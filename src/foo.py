from models.test_model import testNN,testConv
import continual_learning_strategies as cl_strat
import active_learning_strategies as al_strat
from torch.optim import SGD
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split,Subset
import random
from datetime import datetime

num_cycles = 10
init_budget = 5000
cycle_budget = 1000
batch_size = 100
dataset = "MNIST"

'''
fashion_mnist_train = datasets.FashionMNIST('./data', train=True,download=True, transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))

fashion_mnist_test = datasets.MNIST('./data', train=False,download=True, transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))
'''

mnist_train = datasets.MNIST('./data', train=True,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
mnist_test = datasets.MNIST('./data', train=False,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
#train_loader = DataLoader(mnist_train,100,shuffle=True)
test_loader = DataLoader(mnist_test,100,shuffle=True)

model = testNN(28*28,1,400,0.2,0.5,10)
#model = testConv(1,10)
optim = SGD(model.parameters(),lr=0.0035)
#cl_strategy = cl_strat.ElasticWeightConsolidation(model,optim,nn.CrossEntropyLoss(),1.0)
continual_learning_strategy = "ALASSO"
cl_strategy = cl_strat.Alasso(model,optim,nn.CrossEntropyLoss(),0.5,0.7,1.5,0.7)
active_learning_strategy = "LC"
al_strategy = al_strat.LC(model,mnist_train,10,test_loader,batch_size,cycle_budget,init_budget,None)
loaders_dict = {'train': None, 'val': test_loader}

unlabeled_set = [i for i in range(len(mnist_train))]
score_list = []
labeled_set = []
for i in range(init_budget):
    labeled_set.append(random.randint(0,len(mnist_train)-1))
training_set = Subset(mnist_train,labeled_set)
loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
cl_strategy.train(loaders_dict,5,score_list)
unlabeled_set = [i for i in unlabeled_set if i not in labeled_set]

for i in range(num_cycles):
    al_strategy.feed_current_state(i,unlabeled_set,labeled_set)
    print(f'Running cycle {i+1}/{num_cycles}')
    training_examples = al_strategy.query()
    labeled_set += training_examples
    unlabeled_set = [i for i in unlabeled_set if i not in training_examples]
    training_set = Subset(mnist_train,training_examples)
    loaders_dict['train'] = DataLoader(training_set,batch_size,shuffle=True)
    cl_strategy.train(loaders_dict,5,score_list)

with open("data/experiments/results.txt",'a') as f:
    f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'Continual Learning Strategy: {continual_learning_strategy}\n'
            f'Active Learning Strategy: {active_learning_strategy}\n'
            f'Dataset: {dataset}\n'
            f'Accuracy results at the end of each cycle: {score_list}\n'
            f'{"-"* 70}'+ "\n"
        )
