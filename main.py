import argparse
import random
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
import os
import copy
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import ToTensor
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm
import sys
import wandb
wandb.__version__
from utils import *
from dataloader import 


def parse_option():
    parser = argparse.ArgumentParser(description='FoV')
    # basic config
    parser.add_argument('--model', type=str, required=True, default='MyTransformer',
                        help='model name, options: [Autoformer, Transformer, iTransformer]')
    
    
    # data loader
    parser.add_argument('--root_path', type=str, default=f'{os.path.dirname(os.getcwd())}', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='/processed_data', help='data file')
    

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    
    
    
    
