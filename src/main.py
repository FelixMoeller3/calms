#import argparse
import torch
import numpy as np
from utils import config
from torchvision import datasets,transforms

#parser = argparse.ArgumentParser(description='Run experiment for model stealing')
al = ["Random","BALD", "LC"]
cl = ["Naive", "EWC", "MAS", "IMM", "Alasso"]
config.run_cl_al_config("./src/conf/finetuning/Finetune_MAS.yaml")
#for c in cl:
#    config.run_cl_al_config("./src/conf/basic_model_stealing/" + "LC" + "_" + c + ".yaml")
#for a in al:
#    for c in cl:
#        config.run_cl_al_config("./src/conf/basic_model_stealing/" + a + "_" + c + ".yaml")