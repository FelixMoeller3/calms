import yaml
from active_learning_strategies.strategy import Strategy
import active_learning_strategies as al_strats
import continual_learning_strategies as cl_strats
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from model_stealing.msprocess import ModelStealingProcess
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torch
from models import testConv,testNN
from tqdm import tqdm
from datetime import datetime
import time
import os

CONFIG = ["SUBSTITUTE_MODEL", "BATCH_SIZE", "CYCLES", "RESULTS_FILE", "RESULTS_FILE", "TARGET_MODEL", "EPOCHS"]
SUBSTITUTE_MODEL_CONFIG = ["NAME", "DATASET", "AL_METHOD", "CL_METHOD"]
AL_CONFIG = ["NAME", "INIT_BUDGET", "BUDGET","USE_GPU"]
AL_METHODS = ['LC','BALD','Badge','CoreSet', 'Random']
CL_CONFIG = ["NAME", "OPTIMIZER"]
CL_METHODS = ["Alasso", "IMM", "Naive", "EWC", "MAS"]
OPTIMIZERS =["SGD", "ADAM"]
MODELS = ['Resnet18','Resnext50', 'TestConv']
TARGET_MODEL_CONFIG = ['MODEL','DATASET','EPOCHS','OPTIMIZER']
DATASET_NAMES = ["MNIST","FashionMNIST", "CIFAR-10"]
OPTIMIZER_CONFIG = ["NAME", "LR", "MOMENTUM", "WDECAY", "MILESTONES"]

def run_config(config_path: str) -> ModelStealingProcess:
    '''
        Parses and runs the config file located at 'config_path'. 
        The expected structure can be seen in the 'Example_conf.yaml' file in the 'conf' folder.
    '''
    start = time.time()
    with open(config_path,"r") as f:
        yaml_cfg = yaml.safe_load(f)
    check_attribute_presence(yaml_cfg,CONFIG,"config")
    batch_size = yaml_cfg["BATCH_SIZE"]
    target_model = build_target_model(yaml_cfg["TARGET_MODEL"],batch_size)
    al_method,cl_method,train_set,val_set = build_substitute_model(yaml_cfg["SUBSTITUTE_MODEL"],batch_size,classes)
    cycles = yaml_cfg["CYCLES"]
    ms_process = ModelStealingProcess(target_model,al_method,cl_method)
    num_epochs = yaml_cfg["EPOCHS"]
    accuracies = ms_process.steal_model(train_set,val_set,batch_size,cycles,num_epochs)
    duration = time.time() - start
    hours = int(duration)//3600
    minutes = (int(duration) % 3600) // 60
    seconds = int(duration) % 60
    time_string = "{:02}h:{:02}m:{:02}s".format(hours,minutes,seconds)
    os.makedirs(yaml_cfg["RESULTS_FOLDER"],exist_ok=True)
    with open(yaml_cfg["RESULTS_FOLDER"] + yaml_cfg["RESULTS_FILE"],'a+') as f:
        f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} after {time_string}\n'
                f'Target model: {yaml_cfg["TARGET_MODEL"]["MODEL"]}, trained on {yaml_cfg["TARGET_MODEL"]["DATASET"]}\n'
                f'Substitute model: {yaml_cfg["SUBSTITUTE_MODEL"]["NAME"]}, trained on {yaml_cfg["SUBSTITUTE_MODEL"]["DATASET"]}\n'
                f'Continual Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["NAME"]}\n'
                f'Active Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["NAME"]}\n'
                f'Accuracy results at the end of each cycle: {accuracies}\n'
                f'{"-"* 70}'+ "\n"
            )

def run_cl_al_config(config_path: str) -> ModelStealingProcess:
    '''
        Parses and runs the config file located at 'config_path'. 
        The expected structure can be seen in the 'Example_conf.yaml' file in the 'conf' folder.
    '''
    start = time.time()
    with open(config_path,"r") as f:
        yaml_cfg = yaml.safe_load(f)
    check_attribute_presence(yaml_cfg,CONFIG,"config")
    batch_size = yaml_cfg["BATCH_SIZE"]
    al_method,cl_method,train_set,val_set = build_substitute_model(yaml_cfg["SUBSTITUTE_MODEL"],batch_size)
    cycles = yaml_cfg["CYCLES"]
    ms_process = ModelStealingProcess(None,al_method,cl_method)
    num_epochs = yaml_cfg["EPOCHS"]
    accuracies = ms_process.continual_learning(train_set,val_set,batch_size,cycles,num_epochs)
    duration = time.time() - start
    hours = int(duration)//3600
    minutes = (int(duration) % 3600) // 60
    seconds = int(duration) % 60
    time_string = "{:02}h:{:02}m:{:02}s".format(hours,minutes,seconds)
    os.makedirs(yaml_cfg["RESULTS_FOLDER"],exist_ok=True)
    with open(yaml_cfg["RESULTS_FOLDER"] + yaml_cfg["RESULTS_FILE"],'a+') as f:
        f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} after {time_string}\n'
                f'Model: {yaml_cfg["SUBSTITUTE_MODEL"]["NAME"]}, trained on {yaml_cfg["SUBSTITUTE_MODEL"]["DATASET"]}\n'
                f'Continual Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["NAME"]}\n'
                f'Active Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["NAME"]}\n'
                f'Accuracy results at the end of each cycle: {accuracies}\n'
                f'{"-"* 70}'+ "\n"
            )

def run_al_comfig(config_path: str) -> ModelStealingProcess:
    '''
        Parses and runs the config file located at 'config_path'. 
        The expected structure can be seen in the 'Example_conf.yaml' file in the 'conf' folder.
    '''
    start = time.time()
    with open(config_path,"r") as f:
        yaml_cfg = yaml.safe_load(f)
    check_attribute_presence(yaml_cfg,CONFIG,"config")
    batch_size = yaml_cfg["BATCH_SIZE"]
    al_method,cl_method,train_set,val_set = build_substitute_model(yaml_cfg["SUBSTITUTE_MODEL"],batch_size)
    cycles = yaml_cfg["CYCLES"]
    ms_process = ModelStealingProcess(None,al_method,cl_method)
    num_epochs = yaml_cfg["EPOCHS"]
    accuracies = ms_process.active_learning(train_set,val_set,batch_size,cycles,num_epochs)
    duration = time.time() - start
    hours = int(duration)//3600
    minutes = (int(duration) % 3600) // 60
    seconds = int(duration) % 60
    time_string = "{:02}h:{:02}m:{:02}s".format(hours,minutes,seconds)
    os.makedirs(yaml_cfg["RESULTS_FOLDER"],exist_ok=True)
    with open(yaml_cfg["RESULTS_FOLDER"] + yaml_cfg["RESULTS_FILE"],'a+') as f:
        f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} after {time_string}\n'
                f'Model: {yaml_cfg["SUBSTITUTE_MODEL"]["NAME"]}, trained on {yaml_cfg["SUBSTITUTE_MODEL"]["DATASET"]}\n'
                f'Active Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["NAME"]}\n'
                f'Accuracy results at the end of each cycle: {accuracies}\n'
                f'{"-"* 70}'+ "\n"
            )

def check_attribute_presence(config: dict, attributes: list[str],config_name: str) -> None:
    '''
        Checks if all attributes are present in the config dict. The config_name is needed to
        produce an accurate error message.
    '''
    for elem in attributes:
        if elem not in config:
            raise AttributeError(f"{elem} must to be specified in {config_name}")

def build_target_model(target_model_config: dict,batch_size:int) -> nn.Module:
    check_attribute_presence(target_model_config,TARGET_MODEL_CONFIG,"target model config")
    train_set,input_dim,num_classes = load_dataset(target_model_config['DATASET'],True)
    target_model = build_model(target_model_config["MODEL"],input_dim,num_classes)
    train_loader = DataLoader(train_set,batch_size,True)
    val_set,_,_ = load_dataset(target_model_config['DATASET'],False)
    val_loader = DataLoader(val_set,batch_size,True)
    optimizer = build_optimizer(target_model_config["OPTIMIZER"],target_model)
    train_model(target_model,train_loader,val_loader,optimizer,target_model_config["EPOCHS"])
    return target_model

def build_model(name: str, input_dim:tuple[int], num_classes: int) -> nn.Module:
    #TODO: Add models here
    if name == "Resnet18":
        return None
    elif name == "Resnext50":
        return None
    elif name == "TestConv":
        return testConv(input_dim,num_classes)
    else:
        raise AttributeError(f"Model name unknown. Got {name}, but expected one of {','.join(MODELS)}")

def load_dataset(name: str,train:bool) -> tuple[Dataset,torch.Size,int]:
    '''
        Loads a dataset into memory and returns it along with the dimension of a single instance
        and the number of target classes the dataset has.
    '''
    if name == "MNIST":
        dataset = datasets.MNIST("./data",train,transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]),download=True)
    elif name == "FashionMNIST":
        dataset = datasets.FashionMNIST("./data",train,transform=transforms.Compose([
                       transforms.ToTensor()
                   ]),download=True)
    elif name == "CIFAR-10":
        dataset = datasets.CIFAR10("./data",train,transform=transforms.Compose([
                       transforms.ToTensor()
                   ]),download=True)
    else:
        raise AttributeError(f"Dataset unknown. Got {name}, but expected one of {','.join(DATASET_NAMES)}")
    return dataset,dataset[0][0].shape,len(dataset.class_to_idx)

def build_optimizer(config: dict,model:nn.Module) -> torch.optim.Optimizer:
    check_attribute_presence(config,OPTIMIZER_CONFIG,"optimizer config")
    if config["NAME"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),float(config["LR"]),float(config["MOMENTUM"]),weight_decay=float(config["WDECAY"]))
    elif config["NAME"] == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(),float(config["LR"]),weight_decay=float(config["WDECAY"]))
    else:
        raise AttributeError(f"Optimizer unknown. Got {config['NAME']}, but expected one of {','.join(OPTIMIZERS)}")
    return optimizer

def train_model(model: nn.Module,train_loader:DataLoader,val_loader:DataLoader,optimizer: torch.optim.Optimizer,num_epochs:int) -> None:
    criterion = nn.CrossEntropyLoss()
    print("Training model...\n")
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0.0
        for data in tqdm(train_loader,desc=f"Running epoch {epoch+1}/{num_epochs}"):
            inputs,labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        run_val_epoch(model,val_loader,criterion)
    print("Training completed\n")

def run_val_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.CrossEntropyLoss) -> None:
    total_loss = 0.0
    correct_predictions = 0.0
    for data in tqdm(val_loader,desc=f"Computing validation"):
        inputs,labels = data
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        correct_predictions += torch.sum(preds == labels.data).item()
    epoch_loss = total_loss / len(val_loader.dataset)
    epoch_acc = correct_predictions / len(val_loader.dataset)
    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

def build_substitute_model(config: dict,batch_size:int) -> tuple[Strategy,ContinualLearningStrategy,Dataset,Dataset]:
    check_attribute_presence(config,SUBSTITUTE_MODEL_CONFIG,"substitute model config")
    train_set,input_dim,num_classes = load_dataset(config["DATASET"],True)
    substitute_model = build_model(config["NAME"],input_dim,num_classes)
    val_set, _, _ = load_dataset(config["DATASET"],False)
    al_strategy = build_al_strategy(config["AL_METHOD"],substitute_model,train_set,batch_size,num_classes)
    cl_strategy = build_cl_strategy(config["CL_METHOD"],substitute_model)
    return al_strategy,cl_strategy,train_set,val_set


def build_al_strategy(al_config: dict,substitute_model: nn.Module, dataset: Dataset,batch_size:int,num_classes:int) -> Strategy:
    check_attribute_presence(al_config,AL_CONFIG,"active learning config")
    al_config["BATCH"] = batch_size
    al_config["NO_CLASSES"] = num_classes
    if al_config["NAME"] == "BALD":
        al_strat = al_strats.BALD(substitute_model,dataset,**al_config)
    elif al_config["NAME"] == "Badge":
        al_strat = al_strats.Badge(substitute_model,dataset,**al_config)
    elif al_config["NAME"] == "LC":
        al_strat = al_strats.LC(substitute_model,dataset,**al_config)
    elif al_config["NAME"] == "CoreSet":
        al_strat = al_strats.CoreSet(substitute_model,dataset,**al_config)
    elif al_config["NAME"] == "Random":
        al_strat = al_strats.RandomSelection(substitute_model,dataset,**al_config)
    else:
        raise AttributeError(f"Continual learning strategy unknown. Got {al_config['NAME']}, but expected one of {','.join(AL_METHODS)}")
    return al_strat


def build_cl_strategy(cl_config:dict,substitute_model: nn.Module) -> ContinualLearningStrategy:
    check_attribute_presence(cl_config,CL_CONFIG,"continual learning config")
    optimizer = build_optimizer(cl_config["OPTIMIZER"],substitute_model)
    if cl_config["NAME"] == "Alasso":
        cl_strat = cl_strats.Alasso(substitute_model,optimizer,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "IMM":
        cl_strat = cl_strats.IMM(substitute_model,optimizer,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "MAS":
        cl_strat = cl_strats.MAS(substitute_model,optimizer,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "EWC":
        cl_strat = cl_strats.ElasticWeightConsolidation(substitute_model,optimizer,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "Naive":
        cl_strat = cl_strats.Naive(substitute_model,optimizer,nn.CrossEntropyLoss(),**cl_config)
    else:
        raise AttributeError(f"Continual learning strategy unknown. Got {cl_config['NAME']}, but expected one of {','.join(CL_METHODS)}")
    return cl_strat
    
