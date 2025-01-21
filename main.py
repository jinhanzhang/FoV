# Copyright 2020-2024 Jinhan, Xupeng
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
from multiprocessing import context
import random
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
import os
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import ToTensor
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import wandb
wandb.__version__
from utils import *
from dataloader.generate_data import *



def parse_option():
    parser = argparse.ArgumentParser(description='FoV')
    # basic config
    parser.add_argument('--model', type=str, required=True, default='MyTransformer',
                        help='model name, options: [Autoformer, Transformer, iTransformer, Reformer, TimesNet, PatchTST, TimeSeriesTransformerForPrediction]')
    parser.add_argument('--root_path', type=str, default=f'{os.getcwd()}', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='/processed_data', help='data file')
    parser.add_argument('--hist_time', type=float, default=2, help='history data time')
    parser.add_argument('--pred_time', type=float, default=1, help='prediction data time')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--feature_names', type=str, default='XYZ_FEATURE_NAMES', help='[DEFAULT_FEATURE_NAMES, XYZ_FEATURE_NAMES, ONE_FEATURE, SC_FEATURE_NAMES, RPY_FEATURE_NAMES, ANGLE_FEATURE_NAMES]')
    parser.add_argument('--load_model', type=bool, default=False, help='load model')
    parser.add_argument('--use_wandb', type=bool, default=False, help='use wandb for logging')
    # transformer config
    parser.add_argument('--n_heads', type=int, default=3, help='number of heads')
    parser.add_argument('--head_dim', type=int, default=3, help='head dimension')
    parser.add_argument('--d_model', type=int, default=9, help='embedding dimension for time series lib')
    parser.add_argument('--n_decoder_layers', type=int, default=2, help='number of decoder layers')
    parser.add_argument('--n_encoder_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--pe_mode', type=str, default='standard', help='positional encoding mode')
    parser.add_argument('--timestamp', type=int, default=0, help='add additional timestamp feature or not')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--load_data', type=bool, default=True, help='load data or genearate data from dataset')
    parser.add_argument('--train_len', type=int, default=3000, help='train random data length')
    parser.add_argument('--denoise', type=bool, default=False, help='denoise data or not')
    parser.add_argument('--out_suffix', type=str, default='', help='out_suffix')
    parser.add_argument('--dataset', type=str, default='', help='dataset choise:["VVWUB","Shanghai"]')
    parser.add_argument('--loss_func', type=str, default='', help='specify loss function: ["my_loss"]')
    return parser.parse_args()
    

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    id = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    args = parse_option()
    saved_path = f'denoised_saved_results/{args.model}_hist_{args.hist_time}_pred_{args.pred_time}_bs_{args.batch_size}_feat_{args.feature_names}_epoch_{args.num_epochs}_n_heads_{args.n_heads}_head_dim_{args.head_dim}_train_len_{args.train_len}_{args.out_suffix}'
    print(args.denoise)
    if args.denoise:
        saved_path = f'{saved_path}_denoise'
    if args.loss_func=='my_loss':
        saved_path = f'{saved_path}_my_loss'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # parse augments
    
    
    # config
    PROJECT_PATH = args.root_path
    FRAME_RATE = 60 # 60 frames/sec
    MAX_HISTORY_TIME = 10
    MAX_PREDICTION_TIME = 10
    HISTORY_TIME = args.hist_time
    PREDICTION_TIME = args.pred_time
    HISTORY_LENGTH = int(HISTORY_TIME*FRAME_RATE)
    PREDICTION_LENGTH = int(PREDICTION_TIME*FRAME_RATE)
    MAX_HISTORY_LENGTH = MAX_HISTORY_TIME*FRAME_RATE
    MAX_PREDICTION_LENGTH = MAX_PREDICTION_TIME*FRAME_RATE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", DEVICE)
    TOTAL_FEATURE_NAMES = ['head_x','head_y','head_z','head_r_sin','head_r_cos','head_p_sin','head_p_cos','head_y_sin',\
    'head_y_cos','head_rx','head_ry','head_rz']
    DEFAULT_FEATURE_NAMES = ['head_x','head_y','head_z','head_r_sin','head_r_cos','head_p_sin','head_p_cos','head_y_sin',\
    'head_y_cos']
    XYZ_FEATURE_NAMES = ['head_x', 'head_y', 'head_z']
    ONE_FEATURE_X = ['head_x']
    ONE_FEATURE_Y = ['head_y']
    ONE_FEATURE_Z = ['head_z']
    SC_FEATURE_NAMES = ['head_r_sin','head_r_cos','head_p_sin','head_p_cos','head_y_sin','head_y_cos']
    RPY_FEATURE_NAMES = ['head_rx','head_ry','head_rz']
    ANGLE_FEATURE_NAMES = ['head_r_cos','head_p_sin','head_p_cos','head_y_sin','head_y_cos','head_rx','head_ry','head_rz']
    FEATURE_NAMES = eval(args.feature_names)
    FEATURE_INDEX = [TOTAL_FEATURE_NAMES.index(x) for x in FEATURE_NAMES]
    DEFAULT_FEATURE_SIZE = len(DEFAULT_FEATURE_NAMES)
    FEATURE_SIZE = len(FEATURE_NAMES)
    if args.timestamp:
        FEATURE_SIZE += 1
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
    LOAD_MODEL = args.load_model
    TRAIN_LEN = args.train_len
    DENOISE_FLAG = args.denoise
    
    # load data
    
    if DENOISE_FLAG:
        processed_data_path = f'{PROJECT_PATH}/denoised_processed_data'
        processed_long_sequence_path = f'{PROJECT_PATH}/denoised_processed_long_sequence'
        # dataset_path = f'{PROJECT_PATH}/denoised_dataset'
        dataset_path = f'{PROJECT_PATH}/dataset'
        
    else:
        processed_data_path = f'{PROJECT_PATH}/processed_data'
        processed_long_sequence_path = f'{PROJECT_PATH}/processed_long_sequence'  
        dataset_path = f'{PROJECT_PATH}/dataset'
    #import pdb; pdb.set_trace()
    if args.load_data is True and os.path.isfile(f'{processed_data_path}/x_val_{HISTORY_TIME}_{PREDICTION_TIME}.csv'):
        print("load data from stored files")

        # read val and test processed data
        x_val = np.loadtxt(f'{processed_data_path}/x_val_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,HISTORY_LENGTH,DEFAULT_FEATURE_SIZE))
        y_val = np.loadtxt(f'{processed_data_path}/y_val_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,PREDICTION_LENGTH,DEFAULT_FEATURE_SIZE))
        x_test = np.loadtxt(f'{processed_data_path}/x_test_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,HISTORY_LENGTH,DEFAULT_FEATURE_SIZE))
        y_test = np.loadtxt(f'{processed_data_path}/y_test_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((-1,PREDICTION_LENGTH,DEFAULT_FEATURE_SIZE))
        mean_std = np.loadtxt(f'{processed_data_path}/xyz_mean_std_{HISTORY_TIME}_{PREDICTION_TIME}.csv', dtype='float32', delimiter=',').reshape((3, -1))
    else:
        # generate val and test data from dataset
        print ('generate from scratch')
        _,_, x_val, y_val, x_test, y_test, mean_std = generate_data(dataset_path, processed_data_path, HISTORY_TIME, PREDICTION_TIME, FRAME_RATE, DENOISE_FLAG)
        print("generate data: ", x_val.shape, y_val.shape, x_test.shape, y_test.shape, mean_std.shape)
        
    # synthetic data
    # a = random.randint(0, 9)
    # b = random.randint(0, 9)
    # c = random.randint(0, 9)
    # sequence1 = np.linspace(a, a+3, 10000).astype('f')
    # sequence2 = np.linspace(b, b+6, 10000).astype('f')
    # sequence3 = np.linspace(c, c+9, 10000).astype('f')
    # combined_sequence = np.column_stack((sequence1, sequence2, sequence3))
    # x = []
    # y = []
    # for i in range(0,1000,8):
    #     x.append(combined_sequence[i:i+120])
    #     y.append(combined_sequence[i+120:i+240])
    # x = np.array(x)
    # y = np.array(y)
    # print(x.shape, y.shape, x.dtype, y.dtype)
    
    # create dataset and dataloader
    feature_names = FEATURE_NAMES
    feature_idx = FEATURE_INDEX
    # x_train = x_train[:,:,feature_idx]
    # y_train = y_train[:,:,feature_idx]
    x_val = x_val[:,:,feature_idx]
    y_val = y_val[:,:,feature_idx]
    x_test = x_test[:,:,feature_idx]
    y_test = y_test[:,:,feature_idx]
    # train_data = VVWUBDataset(x_train, y_train, feature_idx, timestamp=args.timestamp)
    train_data = VVWUBTrainDataset(dataset_path, processed_long_sequence_path, feature_idx, HISTORY_TIME, PREDICTION_TIME, FRAME_RATE, TRAIN_LEN, timestamp=args.timestamp)
    print("train_data len: ", len(train_data))
    print("train_data len: ", len(train_data[:100]))
    val_data = VVWUBDataset(x_val, y_val, feature_idx, timestamp=args.timestamp)
    test_data = VVWUBDataset(x_test, y_test, feature_idx, timestamp=args.timestamp)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    train_dataloader_viz = DataLoader(train_data, batch_size=100, shuffle=False)
    val_dataloader_viz = DataLoader(val_data, batch_size=80, shuffle=False)
    test_dataloader_viz = DataLoader(test_data, batch_size=80, shuffle=False)
    
    #training hyperparams
    in_seq_len = HISTORY_LENGTH
    out_seq_len = PREDICTION_LENGTH
    feature_size = FEATURE_SIZE
    lr = 0.005
    tf_rate = 0.5
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    n_batches = len(train_dataloader)
    use_wandb = args.use_wandb
    train_std = mean_std[0][3:6]
    print("train_std: ", train_std)
    # init model
    
    model_name = args.model
    config = {
        "model": model_name,
        "history_time": HISTORY_TIME,
        "prediction_time": PREDICTION_TIME,
        "batch_size": BATCH_SIZE,
        "feature_names": FEATURE_NAMES,
        "num_epochs": epochs,
        "denoise_flag": DENOISE_FLAG,
        "loss_func": args.loss_func
    }
    
    if model_name == 'MyTransformer':
        from models.MyTransformer import Transformer
        n_heads = args.n_heads ##4
        head_dim = args.head_dim #32 # dimension of each head, not total
        dim_val =  n_heads*head_dim # embedding dimension, all heads together
        n_decoder_layers = args.n_decoder_layers
        n_encoder_layers = args.n_encoder_layers
        pe_mode = args.pe_mode
        model = Transformer(n_heads, head_dim, feature_size, in_seq_len, out_seq_len, n_encoder_layers, n_decoder_layers, pe_mode, device=DEVICE).to(device=DEVICE)
        additional_config = {"n_heads": n_heads,
        "head_dim": head_dim,
        "dim_val": dim_val,
        "n_decoder_layers": n_decoder_layers,
        "n_encoder_layers": n_encoder_layers,
        "pe_mode": pe_mode}
        config.update(additional_config)
    elif model_name == 'iTransformer':
        sys.path.append('Time-Series-Library-main/')
        sys.path.append('Time-Series-Library-main')
        from time_series_lib.iTransformer import iTransformer
        model = iTransformer(seq_len = in_seq_len, pred_len = out_seq_len, enc_in = FEATURE_SIZE, \
                    d_model = args.d_model,\
                    norm = False).float().to(DEVICE)
        additional_config = {"d_model": args.d_model
        }
        config.update(additional_config)
    elif model_name == 'TimesNet':
        sys.path.append('Time-Series-Library-main/')
        sys.path.append('Time-Series-Library-main')
        from time_series_lib.TimesNet import TimesNet
        print ('in_seq_len',in_seq_len, 'out_seq_len', out_seq_len, 'FEATURE_SIZE', FEATURE_SIZE)
        model = TimesNet(seq_len = in_seq_len, pred_len = out_seq_len, enc_in = FEATURE_SIZE, \
                    d_model =8, c_out = FEATURE_SIZE, \
                    norm = False).float().to(DEVICE)
        #import pdb; pdb.set_trace()
        additional_config = {"d_model": args.d_model
        }
        config.update(additional_config)
    elif model_name == 'PatchTST':
        #TODO: feature dimension issue, PE error
        sys.path.append('Time-Series-Library-main/')
        sys.path.append('Time-Series-Library-main')
        from time_series_lib.PatchTST import PatchTST
        model = PatchTST(seq_len = in_seq_len, pred_len = out_seq_len, enc_in = FEATURE_SIZE, \
                    d_model = 8,\
                    norm = False).float().to(DEVICE)
        additional_config = {"d_model": args.d_model
        }
        config.update(additional_config)
    elif model_name == 'Reformer':
        from reformer_pytorch import Reformer
        n_heads = args.n_heads
        model_r = Reformer(
            dim = FEATURE_SIZE,
            depth = args.n_encoder_layers,
            heads = args.n_heads,
            lsh_dropout = 0.1,
            bucket_size = in_seq_len//2,
            causal = False
        ).to(device=DEVICE).requires_grad_(True)
        
        class Reformer_wrap(nn.Module):
            def __init__(self, in_seq_len, out_seq_len):
                super(Reformer_wrap, self).__init__()
                
                self.seq_len =  in_seq_len
                self.pred_len =  out_seq_len
                self.predict_linear = nn.Linear(
                        self.seq_len,  self.pred_len)

            def forward(self, x):
                out = model_r(x)
                return self.predict_linear(out.permute(0, 2, 1)).permute(
                    0, 2, 1) 
        model = Reformer_wrap(in_seq_len, out_seq_len).to(device=DEVICE).requires_grad_(True)
        
        additional_config = {"n_heads": args.n_heads,
                             "n_encoder_layers": args.n_encoder_layers,
        }
        config.update(additional_config)
        
    elif model_name == 'TimeSeriesTransformerForPrediction':
        from huggingface_hub import hf_hub_download
        from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig, TimeSeriesTransformerModel
        config = TimeSeriesTransformerConfig(
            prediction_length = PREDICTION_LENGTH,
            context_length = HISTORY_LENGTH,
            input_size = FEATURE_SIZE,
            lags_sequence=[1],
            num_dynamic_real_features=FEATURE_SIZE
        )
        model = TimeSeriesTransformerForPrediction(config).to(DEVICE)
        additional_config = {"n_heads": args.n_heads,
                             "n_encoder_layers": args.n_encoder_layers,
                             "n_decoder_layers": args.n_decoder_layers,
                             "pe_mode": args.pe_mode,
                             "loss_func": args.loss_func
        }
        config.update(additional_config)
    else:
        print("Model not found")
        sys.exit(0)

    #init network and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    #keep track of loss for graph
    train_mse_losses = []
    val_mse_losses = []
    test_mse_losses = []
    sep_train_mse_losses = []
    sep_val_mse_losses = []
    sep_test_mse_losses = []
    train_pearsonr_arr = []
    val_pearsonr_arr = []
    test_pearsonr_arr = []
    train_R_arr = []
    val_R_arr = []
    test_R_arr = []
    
    # save the config to result
    # saved_path = f'saved_results/{id}_{model_name}_{HISTORY_TIME}_{PREDICTION_TIME}_batch_{BATCH_SIZE}_feature_{FEATURE_NAMES}_epoch_{EPOCHS}'
    # print(config)
    try:
        with open(f'{saved_path}/config.json', 'w') as f:
            json.dump(config, f)
            print(f'saved config to {saved_path}')
    except (TypeError, OverflowError):
        with open(f'{saved_path}/config.json', 'w') as f:
            config_str = str(config)
            config_str = config_str[config_str.index('{'):]
            config = json.loads(config_str)
            json.dump(config,f)
            print(f'convert from string and saved config to {saved_path}')
    
    
    # Training
    use_wandb = args.use_wandb
    if LOAD_MODEL:
        load_ckpt(f"{PROJECT_PATH}/checkpoints/batch_{BATCH_SIZE}_{HISTORY_TIME}_{PREDICTION_TIME}_ckpts.pt".format(os.getcwd(), epochs, batch_size), model, optimizer)
    lr = 0.1
    # scaler = torch.cuda.amp.GradScaler()
    # print(scaler)
    best_val_loss = float('inf')
    #writer = SummaryWriter("run/loss_plot")
    step = 0
    for epoch in tqdm(range(1, epochs+1)):
        print(f'Epoch #{epoch}')
        epoch_start_time = time.time()
        train_result_path = f'{saved_path}/single_figure/train_epoch_{epoch}'
        os.makedirs(f'{saved_path}/single_figure/',exist_ok=True)
        if epoch % 20 == 0 or epoch == 1:
            train_result_folder = f'{saved_path}/viz/epoch_{epoch}'
            if not os.path.exists(train_result_folder):
                os.makedirs(train_result_folder,exist_ok=True)
        train_loss, sep_train_loss, train_pearsonr, train_R = train(DEVICE, train_result_path, model, config,\
                            train_dataloader, torch.tensor(train_std).to(DEVICE), optimizer, scheduler, step, feature_names, \
                            plot_flag=True if epoch % 20 == 0 else False, timestamp=args.timestamp,\
                            train_dataloader_viz=train_dataloader_viz,train_result_folder=train_result_folder)
    #     print(train_loss)
        train_mse_losses.append(train_loss)
        sep_train_mse_losses.append(sep_train_loss)
        if train_pearsonr is not None:
            train_pearsonr_arr.append(train_pearsonr)
        if train_R is not None:
            train_R_arr.append(train_R)
        mean_loss = sum(train_mse_losses)/len(train_mse_losses)
        val_result_path = f'{saved_path}/single_figure/val_epoch_{epoch}'
        val_result_folder = f'{saved_path}/viz/epoch_{epoch}'
        val_loss, sep_val_loss, val_pearsonr, val_R = validate(DEVICE, val_result_path, model, config,\
                            val_dataloader, torch.tensor(train_std).to(DEVICE), feature_names, plot_flag=True if epoch % 20 == 0 else False,\
                            timestamp=args.timestamp,\
                            val_dataloader_viz=val_dataloader_viz,val_result_folder=val_result_folder,text='val')
        val_mse_losses.append(val_loss.detach().cpu().numpy())
        sep_val_mse_losses.append(sep_val_loss)
        if val_pearsonr is not None:
            val_pearsonr_arr.append(val_pearsonr)
        if val_R is not None:
            val_R_arr.append(val_R)
        test_result_path = f'{saved_path}/single_figure/test_epoch_{epoch}'
        test_result_folder = f'{saved_path}/viz/epoch_{epoch}'
        test_loss, sep_test_loss, test_pearsonr, test_R = validate(DEVICE, test_result_path, model, config,\
                            test_dataloader, torch.tensor(train_std).to(DEVICE), feature_names, plot_flag=True if epoch % 20 == 0 else False,\
                            timestamp=args.timestamp,\
                            val_dataloader_viz=test_dataloader_viz,val_result_folder=test_result_folder,text='test')
        test_mse_losses.append(test_loss.detach().cpu().numpy())
        sep_test_mse_losses.append(sep_test_loss)
        if test_pearsonr is not None:
            test_pearsonr_arr.append(test_pearsonr)
        if test_R is not None:
            test_R_arr.append(test_R)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.4f} | mean loss {mean_loss:8.4f}')
        print('-' * 89)
        if epoch % 20 == 0:
            # plot loss curves
            fig, ax = plt.subplots(3,2, figsize=(12, 12))
            #ax.plot([i.detach().cpu().numpy() for i in test_losses], label='train loss')
            ax[0][0].plot([i for i in val_mse_losses[5:]] , label='validation loss')
            ax[0][0].plot([i  for i in train_mse_losses[5:]] , label='train loss')
            ax[0][0].set_title("Losses")
            ax[0][0].legend()
            ax[0][1].plot([i for i in test_mse_losses], label='test loss')
            ax[0][1].set_title("Losses")
            ax[0][1].legend()
            
            ax[1][0].plot([i for i in val_pearsonr_arr])
            ax[1][0].plot([i  for i in train_pearsonr_arr])
            ax[1][0].set_title("Pearsonr")
            ax[1][0].legend([feature_name+' val pearsonr' for feature_name in feature_names]+[feature_name+' train pearsonr' for feature_name in feature_names])
            ax[1][1].plot([i for i in test_pearsonr_arr])
            ax[1][1].set_title("Pearsonr")
            ax[1][1].legend([feature_name+' test pearsonr' for feature_name in feature_names])
            
            ax[2][0].plot([i for i in val_R_arr])
            ax[2][0].plot([i for i in train_R_arr])
            ax[2][0].set_title("R")
            ax[2][0].legend([feature_name+' validation R' for feature_name in feature_names]+[feature_name+' train R' for feature_name in feature_names])
            ax[2][1].plot([i for i in test_R_arr], label=[feature_name+' test R' for feature_name in feature_names])
            ax[2][1].set_title("R")
            ax[2][1].legend([feature_name+' test R' for feature_name in feature_names])
            
            plt.show()
            fig.savefig(f'{saved_path}/losses_{epoch}.png')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save model
    #         save_ckpt("{}/checkpoints/epoch_{}_ckpts.pt".format(os.getcwd(), epoch), model, optimizer,epochs, val_loss)
            # print("val_mse_losses: ", val_mse_losses)
            save_dict = {
                "epoch": epoch,
                "train_mse_losses": train_mse_losses,
                "val_mse_losses": np.hstack(val_mse_losses).tolist(),
                "test_mse_losses": np.hstack(test_mse_losses).tolist(),
                "sep_train_mse_losses": np.concatenate(sep_train_mse_losses).tolist(),
                "sep_val_mse_losses": np.concatenate(sep_val_mse_losses).tolist(),
                "sep_test_mse_losses": np.concatenate(sep_test_mse_losses).tolist(),
                "train_pearsonr_arr": train_pearsonr_arr,
                "val_pearsonr_arr": val_pearsonr_arr,
                "test_pearsonr_arr": test_pearsonr_arr,
                "train_R_arr": train_R_arr,
                "val_R_arr": val_R_arr,
                "test_R_arr": test_R_arr
            }
            save_ckpt(f"{PROJECT_PATH}/checkpoints/batch_{BATCH_SIZE}_{HISTORY_TIME}to{PREDICTION_TIME}_ckpts.pt", model, optimizer, save_dict)
            config.update(save_dict)
            try:
                with open(f'{saved_path}/config.json', 'w') as f:
                    json.dump(config, f)
                    print(f'saved config to {saved_path}')
            except (TypeError, OverflowError):
                with open(f'{saved_path}/config.json', 'w') as f:
                    config_str = str(config).replace("\'", "\"")
                    # print(config_str)
                    config = json.loads(config_str)
                    json.dump(config,f)
                    print(f'saving metric to config')
        # scheduler.step(mean_loss)
        scheduler.step()
    