#import argparse
import torch
import numpy as np
from utils import config
from torchvision import datasets,transforms
import argparse

parser = argparse.ArgumentParser(description='Run experiment for model stealing')
parser.add_argument("-c","--config",type=str,help="The location of the config file which will be run")
parser.add_argument("-m","--mode",type=str,help="Which mode shall be run. Can be one of [AL,CL,MS]" \
    "for active learning, continual and active learning and model stealing respectively")
args = parser.parse_args()
if args.mode == "AL":
    config.run_al_config(args.config)
elif args.mode == "CL":
    config.run_cl_al_config(args.config)
elif args.mode == "MS":
    config.run_config(args.config)
else:
    raise ValueError(f"Unknown run mode: {args.mode}. Mode must be one of AL,CL,MS")
