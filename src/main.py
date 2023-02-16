#import argparse
import torch
import numpy as np
from utils import config

#parser = argparse.ArgumentParser(description='Run experiment for model stealing')
al = ["Random","BALD", "LC"]
cl = ["Naive", "EWC", "MAS", "IMM", "Alasso"]
for a in al:
    for c in cl:
        config.run_config("./src/conf/" + a + "_" + c + ".yaml")