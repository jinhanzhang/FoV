# Copyright 2020-2024 Xupeng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
from ast import arg, parse
import random
from datetime import datetime
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import wandb
wandb.__version__
from utils import *
# from dataloader import
from sklearn.metrics import mean_squared_error, mean_absolute_error


def parse_option():
    parser = argparse.ArgumentParser(description='FoV')
    # basic config
    parser.add_argument('--model', type=str, required=True, default='XGBOOST',
                        help='model name, options: [Autoformer, Transformer, iTransformer, Reformer, TimesNet, PatchTST]')
    parser.add_argument('--root_path', type=str, default=f'{os.getcwd()}', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='/processed_data', help='data file')
    parser.add_argument('--hist_time', type=int, default=2, help='history data time')
    parser.add_argument('--pred_time', type=int, default=2, help='prediction data time')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--feature_names', type=str, default='XYZ_FEATURE_NAMES', help='[DEFAULT_FEATURE_NAMES, XYZ_FEATURE_NAMES, ONE_FEATURE, SC_FEATURE_NAMES, RPY_FEATURE_NAMES, ANGLE_FEATURE_NAMES]')
    parser.add_argument('--load_model', type=bool, default=False, help='load model')
    parser.add_argument('--use_wandb', type=bool, default=False, help='use wandb for logging')
    # transformer config
    parser.add_argument('--n_heads', type=int, default=3, help='number of heads')
    parser.add_argument('--head_dim', type=int, default=3, help='head dimension')
    parser.add_argument('--dim_val', type=int, default=9, help='embedding dimension')
    parser.add_argument('--n_decoder_layers', type=int, default=2, help='number of decoder layers')
    parser.add_argument('--n_encoder_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--pe_mode', type=str, default='standard', help='positional encoding mode')
    
    
    return parser.parse_args()
    

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    id = datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
    saved_path = f'saved_results/{id}'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # parse augments
    args = parse_option()
    
    # config
    PROJECT_PATH = args.root_path
    FRAME_RATE = 60 # 60 frames/sec
    MAX_HISTORY_TIME = 10
    MAX_PREDICTION_TIME = 10
    HISTORY_TIME = args.hist_time
    PREDICTION_TIME = args.pred_time
    HISTORY_LENGTH = HISTORY_TIME*FRAME_RATE
    PREDICTION_LENGTH = PREDICTION_TIME*FRAME_RATE
    MAX_HISTORY_LENGTH = MAX_HISTORY_TIME*FRAME_RATE
    MAX_PREDICTION_LENGTH = MAX_PREDICTION_TIME*FRAME_RATE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", DEVICE)
    TOTAL_FEATURE_NAMES = ['head_x','head_y','head_z','head_r_sin','head_r_cos','head_p_sin','head_p_cos','head_y_sin',\
    'head_y_cos','head_rx','head_ry','head_rz']
    DEFAULT_FEATURE_NAMES = ['head_x','head_y','head_z','head_r_sin','head_r_cos','head_p_sin','head_p_cos','head_y_sin',\
    'head_y_cos']
    XYZ_FEATURE_NAMES = ['head_x', 'head_y', 'head_z']
    ONE_FEATURE = ['head_x']
    SC_FEATURE_NAMES = ['head_r_sin','head_r_cos','head_p_sin','head_p_cos','head_y_sin','head_y_cos']
    RPY_FEATURE_NAMES = ['head_rx','head_ry','head_rz']
    ANGLE_FEATURE_NAMES = ['head_r_cos','head_p_sin','head_p_cos','head_y_sin','head_y_cos','head_rx','head_ry','head_rz']
    FEATURE_NAMES = eval(args.feature_names)
    FEATURE_INDEX = [TOTAL_FEATURE_NAMES.index(x) for x in FEATURE_NAMES]
    DEFAULT_FEATURE_SIZE = len(DEFAULT_FEATURE_NAMES)
    FEATURE_SIZE = len(FEATURE_NAMES)
    BATCH_SIZE = args.batch_size
    LOAD_MODEL = args.load_model
    
    # load data
    processed_data_path = f'{PROJECT_PATH}/processed_data'
    x_train = np.loadtxt(f'{processed_data_path}/x_train_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,HISTORY_LENGTH,DEFAULT_FEATURE_SIZE))
    y_train = np.loadtxt(f'{processed_data_path}/y_train_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,PREDICTION_LENGTH,DEFAULT_FEATURE_SIZE))
    x_val = np.loadtxt(f'{processed_data_path}/x_val_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,HISTORY_LENGTH,DEFAULT_FEATURE_SIZE))
    y_val = np.loadtxt(f'{processed_data_path}/y_val_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,PREDICTION_LENGTH,DEFAULT_FEATURE_SIZE))
    x_test = np.loadtxt(f'{processed_data_path}/x_test_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,HISTORY_LENGTH,DEFAULT_FEATURE_SIZE))
    y_test = np.loadtxt(f'{processed_data_path}/y_test_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,PREDICTION_LENGTH,DEFAULT_FEATURE_SIZE))
    mean_std = np.loadtxt(f'{processed_data_path}/xyz_mean_std_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((3, -1))
    
    # create dataset and dataloader
    feature_names = FEATURE_NAMES
    feature_idx = FEATURE_INDEX
    x_train = x_train[:,:,feature_idx]
    y_train = y_train[:,:,feature_idx]
    x_val = x_val[:,:,feature_idx]
    y_val = y_val[:,:,feature_idx]
    x_test = x_test[:,:,feature_idx]
    y_test = y_test[:,:,feature_idx]
    train_data = FoVDataset(x_train, y_train, feature_idx)
    val_data = FoVDataset(x_val, y_val, feature_idx)
    test_data = FoVDataset(x_test, y_test, feature_idx)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    
    #training hyperparams
    in_seq_len = HISTORY_LENGTH
    out_seq_len = PREDICTION_LENGTH
    feature_size = FEATURE_SIZE
    lr = 0.005
    tf_rate = 0.5
    epochs = 200
    batch_size = BATCH_SIZE
    n_batches = len(train_dataloader)
    use_wandb = args.use_wandb
    # init model
    model_name = args.model
    if model_name == 'XGBOOST':
        import xgboost as xgb #pip install xgboost
        # https://www.kaggle.com/code/furiousx7/xgboost-time-series
        #TODO: 3 dimensional issue
        #a temporal work around is to flatten the time and feature dimension? i don't think it makes sense
        #should maybe flatten batch and feature dimension
        #should we add another time count dimension as the feature?
        reg = xgb.XGBRegressor(n_estimators=1000)
        bs, input_temporal_dim, input_feature_dim = x_train.shape
        bs, output_temporal_dim, output_feature_dim = y_train.shape
        # x_train = x_train.reshape(x_train.shape[0], -1)
        # x_test = x_test.reshape(x_test.shape[0], -1)
        # y_train = y_train.reshape(y_train.shape[0], -1)
        # y_test = y_test.reshape(y_test.shape[0], -1)
        x_train = np.swapaxes(x_train,2,1).reshape(-1, input_temporal_dim)
        x_test = np.swapaxes(x_test,2,1).reshape(-1, input_temporal_dim)
        y_train = np.swapaxes(y_train,2,1).reshape(-1, output_temporal_dim)
        y_test = np.swapaxes(y_test,2,1).reshape(-1, output_temporal_dim)
        print ('x_train shape: {}, y_train shape: {}, x_test shape: {}, y_test shape: {}'.format(\
                x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        reg.fit( x_train, y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
            verbose=False) # Change verbose to True if you want to see it train
        y_pred = reg.predict(x_test)
        y_test = y_test.reshape(bs, output_temporal_dim, output_feature_dim)
        y_pred = y_pred.reshape(bs, output_temporal_dim, output_feature_dim)
    else:
        print("Model not found")
        sys.exit(0)

    