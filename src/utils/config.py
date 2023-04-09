from typing import Optional
import yaml
from active_learning_strategies.strategy import Strategy
import active_learning_strategies as al_strats
import continual_learning_strategies as cl_strats
from continual_learning_strategies.cl_base import ContinualLearningStrategy
from process_runner import ModelStealingProcess,ClAlProcess
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torch
from models import testConv,ResNet,BasicBlock,Bottleneck,ThiefConvNet,VGG16
from tqdm import tqdm
from datetime import datetime
import time
import os
from data import TinyImageNet,SmallImagenet
import pickle

CONFIG = ["SUBSTITUTE_MODEL", "BATCH_SIZE", "CYCLES", "RESULTS_FILE", "RESULTS_FILE", "TARGET_MODEL", "EPOCHS","RECOVER_STATE","SAVE_STATE","STATE_DIR"]
SUBSTITUTE_MODEL_CONFIG = ["NAME", "DATASET", "AL_METHOD", "CL_METHOD","USE_LABEL","CONTINUAL"]
AL_CONFIG = ["NAME", "INIT_BUDGET", "BUDGET", "LOOKBACK"]
AL_METHODS = ['LC','BALD','Badge','CoreSet', 'Random', 'VAAL']
CL_CONFIG = ["NAME", "OPTIMIZER"]
CL_METHODS = ["Alasso", "IMM", "Naive", "EWC", "MAS", "AGEM", "Replay", "DGR"]
OPTIMIZERS =["SGD", "ADAM"]
MODELS = ['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 'TestConv','ActiveThiefConv2','ActiveThiefConv3','ActiveThiefConv4' , 'VGG16']
TARGET_MODEL_CONFIG = ['MODEL','DATASET','EPOCHS','OPTIMIZER','TARGET_MODEL_FOLDER','TARGET_MODEL_FILE','TRAIN_MODEL']
DATASET_NAMES = ["MNIST","FashionMNIST", "CIFAR-10","TinyImageNet","SmallImageNet"]
OPTIMIZER_CONFIG = ["NAME", "LR", "MOMENTUM", "WDECAY"]

def run_config(config_path: str) -> None:
    '''
        Parses and runs the config file located at 'config_path'. 
        The expected structure can be seen in the 'Example_conf.yaml' file in the 'conf' folder.
    '''
    start = time.time()
    with open(config_path,"r") as f:
        yaml_cfg = yaml.safe_load(f)
    check_attribute_presence(yaml_cfg,CONFIG,"config")
    batch_size = yaml_cfg["BATCH_SIZE"]
    use_gpu = detect_gpu()
    target_model,num_classes,num_channels,_ = build_target_model(yaml_cfg["TARGET_MODEL"],batch_size,use_gpu)
    val_set = yaml_cfg["TARGET_MODEL"]["DATASET"]
    al_method,cl_method,train_set,val_set = build_substitute_model(yaml_cfg["SUBSTITUTE_MODEL"],batch_size,use_gpu,num_classes,val_set,num_channels)
    prev_state = {}
    if yaml_cfg["RECOVER_STATE"]:
        with open(os.path.join(yaml_cfg["STATE_DIR"],"latest_state.pkl"),'rb') as f:
            prev_state = pickle.load(f)
        cl_method.model.load_state_dict(torch.load(os.path.join(yaml_cfg["STATE_DIR"],"model.pth")))
    cycles = yaml_cfg["CYCLES"]
    use_label = yaml_cfg["SUBSTITUTE_MODEL"]["USE_LABEL"]
    num_epochs = yaml_cfg["EPOCHS"]
    print(f'Model stealing with strategies {yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["NAME"]} and {yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["NAME"]}')
    state_dir = None
    if yaml_cfg["SAVE_STATE"]:
        state_dir = yaml_cfg["STATE_DIR"]
    ms_process = ModelStealingProcess(target_model,al_method,cl_method,train_set,val_set,batch_size,cycles,num_epochs,
                                      yaml_cfg["SUBSTITUTE_MODEL"]["CONTINUAL"],num_classes,build_optimizer,yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["OPTIMIZER"],
                                      use_label,state_dir,yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["COLD_START"],use_gpu,yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["INIT_MODE"])
    accuracies,agreements = ms_process.steal_model(**prev_state)
    duration = time.time() - start
    hours = int(duration)//3600
    minutes = (int(duration) % 3600) // 60
    seconds = int(duration) % 60
    time_string = "{:02}h:{:02}m:{:02}s".format(hours,minutes,seconds)
    os.makedirs(yaml_cfg["RESULTS_FOLDER"],exist_ok=True)
    with open(os.path.join(yaml_cfg["RESULTS_FOLDER"], yaml_cfg["RESULTS_FILE"]),'a+') as f:
        f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} after {time_string}\n'
                f'Config File:\n{yaml.dump(yaml_cfg)}'
                f'Target model: {yaml_cfg["TARGET_MODEL"]["MODEL"]}, trained on {yaml_cfg["TARGET_MODEL"]["DATASET"]}\n'
                f'Target model accuracy on validation set: {val_acc if yaml_cfg["TARGET_MODEL"]["TRAIN_MODEL"] else "unknown"}\n'
                f'Substitute model: {yaml_cfg["SUBSTITUTE_MODEL"]["NAME"]}, trained on {yaml_cfg["SUBSTITUTE_MODEL"]["DATASET"]}\n'
                f'Continual Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["NAME"]}\n'
                f'Active Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["NAME"]}\n'
                f'Accuracy results at the end of each cycle: {accuracies}\n'
                f'Model agreement at the end of each cycle: {agreements}\n'
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
    use_gpu = detect_gpu()
    al_method,cl_method,train_set,val_set = build_substitute_model(yaml_cfg["SUBSTITUTE_MODEL"],batch_size,use_gpu)
    prev_state = {}
    if yaml_cfg["RECOVER_STATE"]:
        with open(os.path.join(yaml_cfg["STATE_DIR"],"latest_state.pkl"),'rb') as f:
            prev_state = pickle.load(f)
        cl_method.model.load_state_dict(torch.load(os.path.join(yaml_cfg["STATE_DIR"],"model.pth")))
    cycles = yaml_cfg["CYCLES"]
    num_epochs = yaml_cfg["EPOCHS"]
    is_continual = yaml_cfg["SUBSTITUTE_MODEL"]["CONTINUAL"]
    state_dir = None
    if yaml_cfg["SAVE_STATE"]:
        state_dir = yaml_cfg["STATE_DIR"]
    process_runner = ClAlProcess(al_method,cl_method,train_set,val_set,batch_size,cycles,num_epochs,is_continual,yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["OPTIMIZER"],
                                 build_optimizer,yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["COLD_START"],yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["INIT_MODE"],state_dir)
    print(f'Running{"continual" if is_continual else ""} active learning with strategies {yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["NAME"]} and {yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["NAME"]}')
    accuracies = process_runner.continual_learning(**prev_state)
    duration = time.time() - start
    hours = int(duration)//3600
    minutes = (int(duration) % 3600) // 60
    seconds = int(duration) % 60
    time_string = "{:02}h:{:02}m:{:02}s".format(hours,minutes,seconds)
    os.makedirs(yaml_cfg["RESULTS_FOLDER"],exist_ok=True)
    with open(os.path.join(yaml_cfg["RESULTS_FOLDER"],yaml_cfg["RESULTS_FILE"]),'a+') as f:
        f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} after {time_string}\n'
                f'Config File: {yaml.dump(yaml_cfg)}\n'
                f'Model: {yaml_cfg["SUBSTITUTE_MODEL"]["NAME"]}, trained on {yaml_cfg["SUBSTITUTE_MODEL"]["DATASET"]}\n'
                f'Continual Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["CL_METHOD"]["NAME"]}\n'
                f'Active Learning Strategy: {yaml_cfg["SUBSTITUTE_MODEL"]["AL_METHOD"]["NAME"]}\n'
                f'Accuracy results at the end of each cycle: {accuracies}\n'
                #f'Class distribution for each query: {",".join([str(dist) for dist in query_dists])}\n'
                f'{"-"* 70}'+ "\n"
            )

def run_target_model_config(config_path: str) -> None:
    '''
        Parses and runs the config file located at 'config_path'. 
        The expected structure can be seen in the 'Example_conf.yaml' file in the 'conf' folder.
    '''
    start = time.time()
    with open(config_path,"r") as f:
        yaml_cfg = yaml.safe_load(f)
    check_attribute_presence(yaml_cfg,CONFIG,"config")
    batch_size = yaml_cfg["BATCH_SIZE"]
    use_gpu = detect_gpu()
    print(f"Training target model: {yaml_cfg['TARGET_MODEL']['MODEL']}")
    model,_,_,val_acc = build_target_model(yaml_cfg["TARGET_MODEL"],batch_size,use_gpu)
    os.makedirs(yaml_cfg["TARGET_MODEL"]["TARGET_MODEL_FOLDER"],exist_ok=True)
    torch.save(model, os.path.join(yaml_cfg["TARGET_MODEL"]["TARGET_MODEL_FOLDER"],yaml_cfg["TARGET_MODEL"]["TARGET_MODEL_FILE"]))
    duration = time.time() - start
    hours = int(duration)//3600
    minutes = (int(duration) % 3600) // 60
    seconds = int(duration) % 60
    time_string = "{:02}h:{:02}m:{:02}s".format(hours,minutes,seconds)
    os.makedirs(yaml_cfg["RESULTS_FOLDER"],exist_ok=True)
    with open(os.path.join(yaml_cfg["RESULTS_FOLDER"],yaml_cfg["RESULTS_FILE"]),'a+') as f:
        f.write(f'Run completed at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} after {time_string}\n'
                f'Config File: {yaml.dump(yaml_cfg)}\n'
                f'Target model: {yaml_cfg["TARGET_MODEL"]["MODEL"]}, trained on {yaml_cfg["TARGET_MODEL"]["DATASET"]}\n'
                f'Validation accuracy of the model: {val_acc:.4f}\n'
                f'Model saved at: {yaml_cfg["TARGET_MODEL"]["TARGET_MODEL_FOLDER"] + yaml_cfg["TARGET_MODEL"]["TARGET_MODEL_FILE"]}\n'
                f'{"-"* 70}'+ "\n"
            )

def detect_gpu() -> bool:
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"Using GPU with CUDA Version: {torch.version.cuda}")
    else:
        print(f"Using CPU since no CUDA Version was found")
    return use_gpu

def check_attribute_presence(config: dict, attributes: list[str],config_name: str) -> None:
    '''
        Checks if all attributes are present in the config dict. The config_name is needed to
        produce an accurate error message.
    '''
    for elem in attributes:
        if elem not in config:
            raise AttributeError(f"{elem} must to be specified in {config_name}")

def build_target_model(target_model_config: dict,batch_size:int,use_gpu:bool) -> tuple[nn.Module,int,int,float]:
    check_attribute_presence(target_model_config,TARGET_MODEL_CONFIG,"target model config")
    train_set,input_dim,num_classes = load_dataset(target_model_config['DATASET'],True)
    if not target_model_config["TRAIN_MODEL"]:
        print(f'Loading model located at {target_model_config["TARGET_MODEL_FOLDER"] + target_model_config["TARGET_MODEL_FILE"]}')
        target_model = torch.load(target_model_config["TARGET_MODEL_FOLDER"] + target_model_config["TARGET_MODEL_FILE"])
        # input_dim[0] contains the number of channels of the image
        return target_model,num_classes,input_dim[0],0.0
    print("Training target model from scratch")
    target_model = build_model(target_model_config["MODEL"],input_dim,num_classes,use_gpu)
    train_loader = DataLoader(train_set,batch_size,True)
    val_set,_,_ = load_dataset(target_model_config['DATASET'],False)
    val_loader = DataLoader(val_set,batch_size,True)
    optimizer,scheduler = build_optimizer(target_model_config["OPTIMIZER"],target_model)
    val_acc = train_model(target_model,train_loader,val_loader,optimizer,scheduler,target_model_config["EPOCHS"],use_gpu)
    return target_model,num_classes,input_dim[0],val_acc

def build_model(name: str, input_dim:tuple[int], num_classes: int, use_gpu:bool) -> nn.Module:
    if name == "Resnet18":
        model = ResNet(BasicBlock, [2,2,2,2], input_dim[0], num_classes)
    elif name == "Resnet34":
        model = ResNet(BasicBlock, [3,4,6,3], input_dim[0], num_classes)
    elif name == "Resnet50":
        model = ResNet(Bottleneck, [3,4,6,3], input_dim[0], num_classes)
    elif name == "Resnet101":
        model = ResNet(Bottleneck, [3,4,23,3], input_dim[0], num_classes)
    elif name == "Resnet152":
        model = ResNet(Bottleneck, [3,8,36,3], input_dim[0], num_classes)
    elif name == "TestConv":
        model = testConv(input_dim,num_classes)
    elif name == "ActiveThiefConv2":
        model = ThiefConvNet(num_conv_blocks=2,input_channels=input_dim[0],num_classes=num_classes,input_dim=input_dim[1])
    elif name == "ActiveThiefConv3":
        model = ThiefConvNet(num_conv_blocks=3,input_channels=input_dim[0],num_classes=num_classes,input_dim=input_dim[1])
    elif name == "ActiveThiefConv4":
        model = ThiefConvNet(num_conv_blocks=4,input_channels=input_dim[0],num_classes=num_classes,input_dim=input_dim[1])
    elif name == "VGG16":
        model = VGG16(input_dim[0],num_classes,input_dim[1])
    else:
        raise AttributeError(f"Model name unknown. Got {name}, but expected one of {','.join(MODELS)}")
    if use_gpu:
        model.cuda()
    return model

def load_dataset(name: str,train:bool,num_channels:Optional[int]=None) -> tuple[Dataset,torch.Size,int]:
    '''
        Loads a dataset into memory and returns it along with the dimension of a single instance
        and the number of target classes the dataset has.
    '''
    channel_change_greyscale = [] if num_channels is None or num_channels == 1 else [transforms.Grayscale(num_channels)]
    channel_change_color = [] if num_channels is None or num_channels == 3 else [transforms.Grayscale(num_channels)]
    if name == "MNIST":
        dataset = datasets.MNIST("./data",train,transform=transforms.Compose( 
                [
                       transforms.ToTensor(),
                       transforms.Resize(32),
                       transforms.Normalize((0.1309,), (0.2892,)),
                   ]
                + channel_change_greyscale),download=True)
        num_classes = 10
    elif name == "FashionMNIST":
        dataset = datasets.FashionMNIST("./data",train,transform=transforms.Compose(
               [
                       transforms.ToTensor(),
                       transforms.Resize(32),
                       transforms.Normalize((0.2854,), (0.3384,))
                   ]
               + channel_change_greyscale),download=True)
        num_classes = 10
    elif name == "CIFAR-10":
        augmentation = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
        normalization = [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ]
        transform = augmentation + normalization if train else normalization
        transform += channel_change_color
        dataset = datasets.CIFAR10("./data",train,transform=transforms.Compose(transform),download=True)
        num_classes = 10
    elif name == "CIFAR-100":
        augmentation = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
        normalization = [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
        transform = augmentation + normalization if train else normalization
        transform += channel_change_color
        dataset = datasets.CIFAR100("./data",train,transform=transforms.Compose(transform),download=True)
        num_classes = 100
    elif name == "TinyImageNet":
        augmentation = [
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalization = [
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
                            std=[0.2618, 0.2537, 0.2677])
        ]
        transform = augmentation + normalization if train else normalization
        transform += channel_change_color
        dataset = TinyImageNet("./data",train,transform=transforms.Compose(transform),download=True)
        num_classes = 200
    elif name.startswith("SmallImageNet"):
        augmentation = [
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalization = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4813, 0.4577, 0.4082], 
                            std=[0.2602, 0.2531, 0.2681])
        ]
        transform = augmentation + normalization if train else normalization
        transform += channel_change_color
        batch_num = name[len("SmallImageNet"):]
        if batch_num == "":
            batch_num = 1
        else:
            batch_num = int(batch_num)
        dataset = SmallImagenet(root="./data",batch_num=batch_num,train=train,transform=transforms.Compose(transform))
        num_classes = 1000
    else:
        raise AttributeError(f"Dataset unknown. Got {name}, but expected one of {','.join(DATASET_NAMES)}")
    print(f"Loaded {name} as {'training' if train else 'validation'} set")
    return dataset,dataset[0][0].shape,num_classes

def build_optimizer(config: dict,model:nn.Module) -> tuple[torch.optim.Optimizer,lr_scheduler.MultiStepLR]:
    check_attribute_presence(config,OPTIMIZER_CONFIG,"optimizer config")
    scheduler = None
    if config["NAME"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),float(config["LR"]),float(config["MOMENTUM"]),weight_decay=float(config["WDECAY"]))
        if "MILESTONES" in config:
            scheduler = lr_scheduler.MultiStepLR(optimizer,config["MILESTONES"])
    elif config["NAME"] == "ADAM":
        w_decay = float(config["WDECAY"]) if "WDECAY" in config else None
        optimizer = torch.optim.Adam(model.parameters(),float(config["LR"]),weight_decay=w_decay)
    else:
        raise AttributeError(f"Optimizer unknown. Got {config['NAME']}, but expected one of {','.join(OPTIMIZERS)}")
    return optimizer,scheduler

def train_model(model: nn.Module,train_loader:DataLoader,val_loader:DataLoader,optimizer: torch.optim.Optimizer,scheduler:lr_scheduler._LRScheduler,num_epochs:int,use_gpu:bool) -> float:
    criterion = nn.CrossEntropyLoss()
    print("Training model...\n")
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0.0
        for data in tqdm(train_loader,desc=f"Running epoch {epoch+1}/{num_epochs}"):
            inputs,labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
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
        if scheduler:
            scheduler.step()
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        val_acc = run_val_epoch(model,val_loader,criterion,use_gpu)
    print("Training completed\n")
    return val_acc

def run_val_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.CrossEntropyLoss,use_gpu:bool) -> float:
    total_loss = 0.0
    correct_predictions = 0.0
    for data in tqdm(val_loader,desc=f"Computing validation"):
        inputs,labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        correct_predictions += torch.sum(preds == labels.data).item()
    epoch_loss = total_loss / len(val_loader.dataset)
    epoch_acc = correct_predictions / len(val_loader.dataset)
    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_acc

def build_substitute_model(config: dict,batch_size:int,use_gpu:bool,num_classes:Optional[int]=None,val_set:Optional[str]=None,num_channels:Optional[str]=None) -> tuple[Strategy,ContinualLearningStrategy,Dataset,Dataset]:
    check_attribute_presence(config,SUBSTITUTE_MODEL_CONFIG,"substitute model config")
    train_set,input_dim,num_classes_new = load_dataset(config["DATASET"],True,num_channels)
    if num_classes is None:
        num_classes = num_classes_new
    substitute_model = build_model(config["NAME"],input_dim,num_classes,use_gpu)
    if val_set is not None:
        val_set, _, _ = load_dataset(val_set,False)
    else:
        val_set, _, _ = load_dataset(config["DATASET"],False,num_channels)
    al_strategy = build_al_strategy(config["AL_METHOD"],substitute_model,train_set,batch_size,num_classes,use_gpu)
    cl_strategy = build_cl_strategy(config["CL_METHOD"],substitute_model,use_gpu)
    return al_strategy,cl_strategy,train_set,val_set


def build_al_strategy(al_config: dict,substitute_model: nn.Module, dataset: Dataset,batch_size:int,num_classes:int,use_gpu:bool) -> Strategy:
    check_attribute_presence(al_config,AL_CONFIG,"active learning config")
    al_config["BATCH"] = batch_size
    al_config["NO_CLASSES"] = num_classes
    al_config["USE_GPU"] = use_gpu
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
    elif al_config["NAME"] == "VAAL":
        al_strat = al_strats.VAAL(substitute_model,dataset,**al_config)
    else:
        raise AttributeError(f"Continual learning strategy unknown. Got {al_config['NAME']}, but expected one of {','.join(AL_METHODS)}")
    return al_strat


def build_cl_strategy(cl_config:dict,substitute_model: nn.Module,use_gpu) -> ContinualLearningStrategy:
    check_attribute_presence(cl_config,CL_CONFIG,"continual learning config")
    optimizer,scheduler = build_optimizer(cl_config["OPTIMIZER"],substitute_model)
    cl_config["USE_GPU"] = use_gpu
    if cl_config["NAME"] == "Alasso":
        cl_strat = cl_strats.Alasso(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "IMM":
        cl_strat = cl_strats.IMM(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "MAS":
        cl_strat = cl_strats.MAS(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "EWC":
        cl_strat = cl_strats.ElasticWeightConsolidation(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "Naive":
        cl_strat = cl_strats.Naive(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "AGEM":
        cl_strat = cl_strats.AGem(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "Replay":
        cl_strat = cl_strats.Replay(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    elif cl_config["NAME"] == "DGR":
        cl_strat = cl_strats.DeepGenerativeReplay(substitute_model,optimizer,scheduler,nn.CrossEntropyLoss(),**cl_config)
    else:
        raise AttributeError(f"Continual learning strategy unknown. Got {cl_config['NAME']}, but expected one of {','.join(CL_METHODS)}")
    return cl_strat
    
