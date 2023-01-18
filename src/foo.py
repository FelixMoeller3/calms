from models.test_model import testNN
from continual_learning_strategies.mas import MAS
from torch.optim import SGD

model = testNN(100,10,8,2)
optim = SGD(model.parameters(),lr=0.01)
mas_strat = MAS(model,optim,None)
print(optim.param_groups)

#for key in mas_strat.omegas:
#    print(mas_strat.omegas[key])
