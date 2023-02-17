#import argparse
import torch
import numpy as np
from utils import config

#parser = argparse.ArgumentParser(description='Run experiment for model stealing')
al = ["Random","BALD", "LC"]
cl = ["Naive", "EWC", "MAS", "IMM", "Alasso"]
for a in al:
    config.run_al_comfig("./src/conf/" + a + "_Naive.yaml")