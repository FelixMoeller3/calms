from models.test_model import testNN
import continual_learning_strategies as cl_strat
import active_learning_strategies as al_strat
from torch.optim import SGD
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split,Subset
import random
from datetime import datetime

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
optim = SGD(model.parameters(),lr=0.01)
#cl_strategy = cl_strat.ElasticWeightConsolidation(model,optim,nn.CrossEntropyLoss(),1.0)
continual_learning_strategy = "ALASSO"
cl_strategy = cl_strat.Alasso(model,optim,nn.CrossEntropyLoss(),1.0,0.7,1.5,0.7)
active_learning_strategy = "LC"
al_strategy = al_strat.LC(model,mnist_train,10,test_loader,32,100,100,None)
loaders_dict = {'train': None, 'val': test_loader}
num_cycles = 10

score_list = []
first_set = []
for i in range(5000):
    first_set.append(random.randint(0,len(mnist_train)-1))
training_set = Subset(mnist_train,first_set)
loaders_dict['train'] = DataLoader(training_set,100,shuffle=True)
cl_strategy.train(loaders_dict,5,score_list)

for i in range(num_cycles):
    print(f'Running cycle {i+1}/{num_cycles}')
    training_examples = al_strategy.query()
    training_set = Subset(mnist_train,training_examples)
    loaders_dict['train'] = DataLoader(training_set,100,shuffle=True)
    cl_strategy.train(loaders_dict,5,score_list)

with open("data/experiments/results.txt",'a') as f:
    f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'Continual Learning Strategy: {continual_learning_strategy}\n'
            f'Active Learning Strategy: {active_learning_strategy}\n'
            f'Accuracy results at the end of each cycle: {score_list}\n'
        )
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