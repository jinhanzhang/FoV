from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import random
from datetime import datetime, date
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import time
import wandb
from dataloader.generate_data import *
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
class FoVDataset(Dataset):
    def __init__(self, x_data, y_data, feature_idx, timestamp=False):
        self.feature_idx = feature_idx
        self.x_data = x_data[:,:,feature_idx]
        self.y_data = y_data[:,:,feature_idx]
        if timestamp:
            bs, input_temporal_dim, input_feature_dim = self.x_data.shape
            bs, output_temporal_dim, output_feature_dim = self.y_data.shape
            timestamp_input = np.concatenate(([ np.arange(input_temporal_dim).reshape(-1,1)[np.newaxis]  \
                    for bs in range(bs)])).astype('float32')
            timestamp_output = np.concatenate(([np.arange(input_temporal_dim, input_temporal_dim+output_temporal_dim).reshape(-1,1)[np.newaxis]  \
                    for bs in range(bs)])).astype('float32')
            self.x_data = np.concatenate((self.x_data, timestamp_input),axis=-1)
            self.y_data = np.concatenate((self.y_data, timestamp_output),axis=-1)
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx])
        y = torch.tensor(self.y_data[idx])
        return x,y
    
class FoVTrainDataset(Dataset):
    def __init__(self, data_path, processed_long_sequence_path, feature_idx,hist_time, pred_time, frame_rate, train_len, timestamp=False):
        self.feature_idx = feature_idx
        self.train_files = glob.glob(processed_long_sequence_path + f'/*_{hist_time}_{pred_time}.csv')
        self.hist_length = int(hist_time*frame_rate)
        self.pred_length = int(pred_time*frame_rate)
        self.total_length = int(self.hist_length + self.pred_length)
        self.timestamp = timestamp
        self.train_len = train_len
        # check if processed data exists
        if len(self.train_files) == 0:
            createAndSaveLongSequence(data_path,processed_long_sequence_path, hist_time, pred_time, frame_rate)
            self.train_files = glob.glob(processed_long_sequence_path + f'/*_{hist_time}_{pred_time}.csv')
        
    def __len__(self):
        return self.train_len
    
    def __getitem__(self, idx):
        file_idx = random.randint(0, len(self.train_files)-1)
        file = self.train_files[file_idx]
        df = pd.read_csv(file, dtype=np.float32)
        start_idx = random.randint(0, len(df)-self.total_length)
        
        x = torch.tensor(df.iloc[start_idx:start_idx+self.hist_length,self.feature_idx].values)
        y = torch.tensor(df.iloc[start_idx+self.hist_length:start_idx+self.hist_length+self.pred_length,self.feature_idx].values)
        return x,y
    
    
# visualize data, target, and prediction
def visualize_data(path, data, target, prediction=None):
    batch_size, in_seq_len, feature_size = data.shape
    _, out_seq_len, _ = target.shape
    rand_batch = random.randint(0, batch_size-1)
    rand_data = data[rand_batch]
    rand_target = target[rand_batch]
    if prediction is not None:
        rand_pred = prediction[rand_batch]
    x = np.arange(out_seq_len)
    plt.figure()
    #import pdb;pdb.set_trace()
    if feature_size<=3:
        fig, ax = plt.subplots(1, feature_size, figsize=(feature_size*4,4))
        ax = [ax] if feature_size==1 else ax
        for i in range(feature_size):
            ax[i].plot(rand_data[:,i])
            ax[i].plot(x+in_seq_len, rand_target[:,i])
            if prediction is not None:
                ax[i].plot(x+in_seq_len, rand_pred[:,i])
    else:
        rows = (feature_size-1)//3+1
        fig, ax = plt.subplots(rows, 3, figsize=(12,4*rows))
        for i in range(rows):
            for j in range(3):
                if i*3+j<feature_size:
                    ax[i][j].plot(rand_data[:,3*i+j])
                    ax[i][j].plot(x+in_seq_len, rand_target[:,3*i+j])
                    if prediction is not None:
                        ax[i][j].plot(x+in_seq_len,rand_pred[:,3*i+j])
    if prediction is None:
        fig.legend(["data","target"])
    else:
        fig.legend(["data","target","pred"])
    plt.show()
    fig.savefig(path)

def visualize_data_all(path, data, target, prediction=None,bs=100,text='train'):
    batch_size, in_seq_len, feature_size = data.shape
    _, out_seq_len, _ = target.shape
    #rand_batch = random.randint(0, batch_size-1)
    #rand_data = data[rand_batch]
    #rand_target = target[rand_batch]
    #if prediction is not None:
    #    rand_pred = prediction[rand_batch]
    #print ('start plot all',text)
    x = np.arange(out_seq_len)
    bs = min(bs,batch_size)
    sample_nums = bs
    for batch in range(bs // sample_nums):
        start = batch * sample_nums
        end = min(batch * sample_nums+sample_nums,bs)
    
        fig, ax = plt.subplots(sample_nums,3,figsize=(5*feature_size,4*sample_nums))
        for j, sample_id in enumerate(np.arange(start, end)):
            for i in range(feature_size):  
                ax[j, i ].plot(np.arange(0,in_seq_len), data[sample_id,:,i ],label='input')
                ax[j, i ].plot(np.arange(in_seq_len, in_seq_len+out_seq_len),target[sample_id,:,i ],label='target')
                if prediction is not None:
                    ax[j, i ].plot(np.arange(in_seq_len, in_seq_len+out_seq_len),prediction[sample_id,:,i ],label='prediction')
                ax[j, i ].legend( )
                #import pdb;pdb.set_trace()
                cc = pearsonr(prediction[sample_id,:,i ],target[sample_id,:,i ])[0]
                mse = mean_squared_error(prediction[sample_id,:,i ],target[sample_id,:,i ])
                ax[j, i ].set_title('sample: {}, CC: {:.03f}, \n mse: {:.03f} feature: {}'.format(sample_id,cc,mse,['head_x','head_y','head_z','head_r_sin','head_r_cos','head_p_sin','head_p_cos','head_y_sin',\
    'head_y_cos','head_rx','head_ry','head_rz'][i]))
        if prediction is None:
            fig.legend(["data","target"])
        else:
            fig.legend(["data","target","pred"])
        ccs = []
        mses = []
        for i in range(feature_size):
            cc = pearsonr(prediction[:,:,i ].ravel(),target[:,:,i ].ravel())[0]
            mse = mean_squared_error(prediction[:,:,i ].ravel(),target[:,:,i ].ravel())
            ccs.append('{:.03f}'.format(cc))
            mses.append('{:.03f}'.format(mse))
        #fig.suptitle('CC: {}, mse: {}'.format(' '.join(ccs),' '.join(mses)))
        #TODO: save mses and ccs of each sample and save to txt or npy
        plt.tight_layout()
        #print ('debug save fig ',text)
        fig.savefig(path+'/{}_{}_{}'.format(text,start,end))

# checkpointing
def save_ckpt(path, model, optimizer, save_dict):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **save_dict
            }, path)

def load_ckpt(path, model, optimizer, device='cuda'):
    if device == 'cpu':
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    del checkpoint['model_state_dict']
    del checkpoint['optimizer_state_dict']
    model.eval()
    return model, optimizer, checkpoint



    
def get_data(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length
    
    t = torch.zeros(batch_size,1).uniform_(0,20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size,1) + t
    
    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:,-output_sequence_length:]

def my_loss(output, target, feature_names):
    # compute loss for each feature
    # output/target: [N, seq_len, 9]
    # change sin cos back to angle ?
    if 'head_r_sin' in feature_names:
        feature_idx = feature_names.index('head_r_sin')
        output_rx = torch.atan2(output[:,:,feature_idx], output[:,:,feature_idx+1]).unsqueeze(2)
        target_rx = torch.atan2(target[:,:,feature_idx], target[:,:,feature_idx+1]).unsqueeze(2)
        output = torch.cat((output,output_rx),-1)
    if 'head_p_sin' in feature_names:
        feature_idx = feature_names.index('head_p_sin')
        output_ry = torch.atan2(output[:,:,feature_idx], output[:,:,feature_idx+1]).unsqueeze(2)
        target_ry = torch.atan2(target[:,:,feature_idx], target[:,:,feature_idx+1]).unsqueeze(2)
        output = torch.cat((output,output_ry),-1)
    if 'head_y_sin' in feature_names:
        feature_idx = feature_names.index('head_y_sin')
        output_rz = torch.atan2(output[:,:,feature_idx], output[:,:,feature_idx+1]).unsqueeze(2)
        target_rz = torch.atan2(target[:,:,feature_idx], target[:,:,feature_idx+1]).unsqueeze(2)
        output = torch.cat((output,output_rz),-1)
        
    
    mse_loss = torch.mean((output - target) ** 2, [0,1])
    pearsonr = stats.pearsonr(output.detach().cpu().flatten(), target.detach().cpu().flatten())
    return mse_loss, pearsonr


def train(device, result_path, model: nn.Module, data_loader, optimizer, scheduler, step, \
            feature_names, plot_flag = False, timestamp=False,train_dataloader_viz=None,\
                train_result_folder=None):
    progress_bar = tqdm(data_loader)
    model.train() # turn on train mode
    feature_size = len(feature_names)
    num_batches = len(data_loader)
    log_interval = num_batches // 5
    loss_names = list(map(lambda x: x+'_loss', feature_names))
    start_time = time.time()
    total_loss = 0.
    return_loss = 0.
    sep_return_loss = np.zeros((1,feature_size if timestamp==False else feature_size+1))
    loss_dict_train = {}
    for batch_idx, (data, targets) in enumerate(progress_bar):
#         print("batch index: ", batch_idx)
        data = data.to(device)
        targets = targets.to(device)
        # with torch.cuda.amp.autocast():
        output = model(data)
        optimizer.zero_grad()
        sep_mse_loss, train_pearsonr = my_loss(output, targets, feature_names)
        loss = torch.sum(sep_mse_loss[:feature_size])
        if model.__class__.__name__ =='Reformer':
            loss.requires_grad_(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        step += 1
        total_loss += loss.item()
        return_loss += loss.item()
        sep_return_loss += sep_mse_loss.detach().cpu().numpy()
        progress_bar.set_postfix_str(f"training loss={loss.item():.4e}|avg training loss={total_loss/(batch_idx+1):.4e}")
        loss_dict_train['training loss'] = loss.item()
        for name_count, loss_name in enumerate(loss_names):
            loss_dict_train[loss_name] = sep_mse_loss[name_count]
        loss_dict_train['avg training loss'] = total_loss/(batch_idx+1)
        loss_dict_train['train pearsonr'] = train_pearsonr
        
        # if use_wandb:
        #     wandb.log(loss_dict_train) 
        if batch_idx % log_interval == 0 and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'{batch_idx:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.5f} | '
                  f'loss {cur_loss:5.5f}')
            total_loss = 0
            start_time = time.time()
            #if plot_flag == True:
                # training prediction
                #print ('training prediction')
                #visualize_data(result_path, data.detach().cpu().numpy(), targets.detach().cpu().numpy(), output.detach().cpu().numpy())
    if plot_flag == True:
        print ('training prediction')
        visualize_data(result_path, data.detach().cpu().numpy(), targets.detach().cpu().numpy(), output.detach().cpu().numpy())
        for batch_idx_viz, (data_viz, targets_viz) in enumerate(train_dataloader_viz):
            output_viz = model(data_viz.to(device))
        visualize_data_all(train_result_folder, data_viz.detach().numpy(), targets_viz.detach().cpu().numpy(), output_viz.detach().cpu().numpy())
    return return_loss/(batch_idx+1), sep_return_loss/(batch_idx+1), train_pearsonr

def validate(device, result_path, model: nn.Module, dataloader: DataLoader, feature_names,\
    plot_flag = False, timestamp=False,val_dataloader_viz=None,\
                val_result_folder=None,text='val'):
    print ('validating', text)
    feature_size = len(feature_names)
    loss_names = list(map(lambda x: x+'_loss', feature_names))
    model.eval()
    total_loss = 0.
    sep_total_loss = np.zeros((1,feature_size if timestamp==False else feature_size+1))
    loss_dict_valid = {}
    iter_count = 0
    with torch.no_grad():
        for (data, targets) in dataloader:
            data = data.to(device) # [N, seq_len, feature_size]
            targets = targets.to(device) # [N, seq_len, feature_size]
            output = model(data)
            #if plot_flag and iter_count == 0:
            #    print ('validation prediction')
            #    visualize_data(result_path, data.detach().cpu().numpy(), targets.detach().cpu().numpy(), output.detach().cpu().numpy())
                                
                # if use_wandb:
                #     wandb.log({"validation plot": fig})
            sep_valid_loss, valid_pearsonr = my_loss(output, targets, feature_names)
            total_loss += torch.sum(sep_valid_loss)
            sep_total_loss += sep_valid_loss.detach().cpu().numpy()
            #import pdb;pdb.set_trace()
            for name_count, loss_name in enumerate(loss_names):
                loss_dict_valid[loss_name] = sep_valid_loss[name_count]
            #loss_dict_valid['valid loss'] = sep_valid_loss
            loss_dict_valid['valid pearsonr'] = valid_pearsonr[0]
            iter_count += 1
            # if use_wandb:
            #     wandb.log(loss_dict_valid) 
    if plot_flag == True:
        print ('validation prediction save viz all',text)
        visualize_data(result_path, data.detach().cpu().numpy(), targets.detach().cpu().numpy(), output.detach().cpu().numpy())        
        for batch_idx_viz, (data_viz, targets_viz) in enumerate(val_dataloader_viz):
            output_viz = model(data_viz.to(device))
        visualize_data_all(val_result_folder, data_viz.detach().numpy(), \
            targets_viz.detach().cpu().numpy(), output_viz.detach().cpu().numpy(),bs=80, text=text)

    return total_loss/(len(dataloader) - 1), sep_total_loss/(len(dataloader) - 1), valid_pearsonr
    

# def inference(model: nn.Module, dataloader: DataLoader) -> float:
#     model.eval()  # turn on evaluation mode
#     total_loss = 0.
#     o = []
#     loss_dict_test = {}
#     with torch.no_grad():
#         for (data, targets) in dataloader:
#             data = data.to(device=DEVICE) # [N, seq_len, feature_size]
#             targets = targets.to(device=DEVICE) # [N, seq_len, feature_size]
#             N, seq_len, feature_size = targets.shape
#             dec_input = torch.zeros(targets.shape)
#             for i in range(seq_len):
#                 if i == 0:
#                     dec_input = data
#                     output = model(data, dec_input)
#                     o = output
#                 else:
#                     dec_input[:,:seq_len-i,:] = data[:,i:,:]
#                     dec_input[:,seq_len-i:,:] = o
#                     output = model(data, dec_input)
#                     o = torch.hstack(o, output[:,-1,:])
#             test_loss = my_loss(o, targets).item()
#             total_loss += test_loss
#             loss_dict_test['test loss'] = test_loss
#             wandb.log(loss_dict_test) 
#             # print("val loss: ", total_loss)
#             # data = data.permute(1, 0, 2) #.detach().cpu().numpy()
#             # output = output.permute(1, 0, 2) #.detach().cpu().numpy()
#             # print("output2: ", output[0][0])
#     return total_loss / (len(dataloader) - 1)