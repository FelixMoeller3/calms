import yaml
from active_learning_strategies.strategy import Strategy
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from model_stealing.msprocess import ModelStealingProcess
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torch
from tqdm import tqdm

CONFIG = ["SUBSTITUTE_MODEL", "BATCH_SIZE", "CYCLES", "RESULTS_FOLDER", "TARGET_MODEL"]
SUBSTITUTE_MODEL_CONFIG = ["NAME", "DATASET", "AL_METHOD", "CL_METHOD"]
AL_CONFIG = ["NAME", "INIT_BUDGET", "BUDGET"]
AL_METHODS = ['LC','BALD','Badge','CoreSet', 'random']
CL_CONFIG = ["NAME", "OPTIMIZER"]
CL_METHODS = ["Alasso", "IMM", "Naive", ""]
MODELS = ['Resnet18','Resnext50']
TARGET_MODEL_CONFIG = ['MODEL','DATASET','EPOCHS','OPTIMIZER']
DATASET_NAMES = ["MNIST","FashionMNIST"]
OPTIMIZER_CONFIG = ["NAME", "LR", "MOMENTUM", "WDECAY", "MILESTONES"]

def parse_config(config_path: str) -> ModelStealingProcess:
    '''
        Parses the config file located at 'config_path'. 
        The expected structure can be seen in the 'Example_conf.yaml' file in the 'conf' folder.
    '''
    with open(config_path,"r") as f:
        yaml_cfg = yaml.safe_load(f)
    check_attribute_presence(yaml_cfg,CONFIG,"config")
    batch_size = yaml_cfg["BATCH_SIZE"]
    target_model = build_target_model(yaml_cfg["TARGET_MODEL"],batch_size)
    al_method,cl_method,dataset = build_substitute_model(yaml_cfg["SUBSTITUTE_MODEL"],batch_size)
    
    ms_process = ModelStealingProcess(target_model,al_method,cl_method)
    return ms_process,dataset

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
    target_model = build_model(target_model_config["MODEL"])
    train_set = load_dataset(target_model_config['DATASET'],batch_size,True)
    train_loader = DataLoader(train_set,batch_size,True)
    optimizer = build_optimizer(target_model_config["OPTIMIZER"])
    train_model(target_model,train_loader,optimizer,target_model_config["EPOCHS"])
    return target_model

def build_model(name: str) -> nn.Module:
    #TODO: Add models here
    if name == "Resnet18":
        return None
    elif name == "Resnext50":
        return None
    else:
        raise AttributeError(f"Model name unknown. Got {name}, but expected one of {','.join(MODELS)}")

def load_dataset(name: str,train:bool) -> Dataset:
    if name == "MNIST":
        dataset = datasets.MNIST("./data",train,transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]),download=True)
    elif name == "FashionMNIST":
        dataset = datasets.FashionMNIST("./data",train,transform=transforms.Compose([
                       transforms.ToTensor()
                   ]),download=True)
    else:
        raise AttributeError(f"Dataset unknown. Got {name}, but expected one of {','.join(DATASET_NAMES)}")
    return dataset

def build_optimizer(config: dict,model:nn.Module) -> torch.optim.Optimizer:
    check_attribute_presence(config,OPTIMIZER_CONFIG,"optimizer config")
    if config["NAME"] == "SGD":
        torch.optim.SGD(model.parameters(),config["LR"],config["MOMENTUM"],weight_decay=config["WDECAY"])
    else:
        raise AttributeError(f"Optimizer unknown. Got {config['NAME']}, but expected one of {','.join(DATASET_NAMES)}")

def train_model(model: nn.Module,dataloader:DataLoader,optimizer: torch.optim.Optimizer,num_epochs:int) -> None:
    criterion = nn.CrossEntropyLoss()
    print("Training model...\n")
    total_loss = 0.0
    correct_predictions = 0.0
    for epoch in num_epochs:
        total_loss = 0.0
        for data in tqdm(dataloader,desc=f"Running epoch {epoch+1}/{num_epochs}:"):
            inputs,labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels.data).item()
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions / len(dataloader.dataset)
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print("Training completed\n")

def build_substitute_model(config: dict,batch_size:int) -> tuple[Strategy,ContinualLearningStrategy,Dataset]:
    check_attribute_presence(config,SUBSTITUTE_MODEL_CONFIG,"substitute model config")
    substitute_model = build_model(config["NAME"])
    train_set = load_dataset(config["DATASET"],batch_size,True)
    al_strategy = build_al_strategy(config["AL_METHOD"],substitute_model)
    cl_strategy = build_cl_strategy(config["CL_METHOD"],substitute_model)
    return al_strategy,cl_strategy,train_set


def build_al_strategy(yaml_cfg: dict,substitute_model: nn.Module) -> Strategy:
    if 'AL_METHOD' not in yaml_cfg:
        raise AttributeError("Active learning method must to be specified in config")
    al_config = yaml_cfg['AL_METHOD']
    for attribute in ['DATASET','NO_CLASSES','batch','budget','init_budget']:
        if attribute not in al_config:
            raise AttributeError(f"{attribute} must to be specified for AL_METHOD")
    if al_config["NAME"] not in AL_METHODS:
        raise AttributeError(f"Active learning method unknown. Got {yaml_cfg['METHOD']}, but expected one of {'.'.join(AL_METHODS)}")
    
    al_params = {'model': substitute_model, 'data_unlabeled': None}
    al_params

    # Model will be set later 
    al_params['model'] = None
    
        
    if yaml_cfg['AL_METHOD'] == 'LC':
        pass

def build_cl_strategy(cl_config:dict,substitute_model: nn.Module) -> ContinualLearningStrategy:
    check_attribute_presence(cl_config,)
    cl_config = yaml_cfg["CL_METHOD"]
    #if yaml_cfg[""]
