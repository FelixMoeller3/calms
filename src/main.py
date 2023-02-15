#import argparse
import torch
import numpy as np

#parser = argparse.ArgumentParser(description='Run experiment for model stealing')

a = torch.tensor([[1,2,3]])

print(torch.max(a,1))