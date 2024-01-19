import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader


class FoVDataset(Dataset):
    def __init__(self, x_data, y_data, feature_idx):
        self.feature_idx = feature_idx
        self.x_data = x_data[:,:,feature_idx]
        self.y_data = y_data[:,:,feature_idx]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx])
        y = torch.tensor(self.y_data[idx])
        return x,y
    
# visualize data, target, and prediction
def visualize_data(data, target, prediction=None):
    batch_size, seq_len, feature_size = data.shape
    rand_batch =random.randint(batch_size)
    rand_data = data[rand_batch]
    rand_target = target[rand_batch]
    if prediction is not None:
        rand_pred = prediction[rand_batch]
    x = np.arange(seq_len)
    plt.figure()
    if feature_size<=3:
        fig, ax = plt.subplots(1, feature_size, figsize=(feature_size*4,4))
        for i in range(feature_size):
            ax[i].plot(rand_data[:,i])
            ax[i].plot(x+seq_len, rand_target[:,i])
            if prediction is not None:
                ax[i].plot(x+seq_len, rand_pred[:,i])
    else:
        rows = (feature_size-1)//3+1
        fig, ax = plt.subplots(rows, 3, figsize=(12,4*rows))
        for i in range(rows):
            for j in range(3):
                if i*3+j<feature_size:
                    ax[i][j].plot(rand_data[:,3*i+j])
                    ax[i][j].plot(x+seq_len, rand_target[:,3*i+j])
                    if prediction is not None:
                        ax[i][j].plot(x+seq_len,rand_pred[:,3*i+j])
    if prediction is None:
        fig.legend(["data","target"])
    else:
        fig.legend(["data","target","pred"])
    plt.show()


# checkpointing
def save_ckpt(path, model, optimizer, epoch, train_losses, val_losses):
    
#     torch.save({'net': network.state_dict(), 'opt': optimizer.state_dict()}, fn)
    path = f'{path}/batch_{BATCH_SIZE}_{HISTORY_TIME}_{PREDICTION_TIME}_ckpts.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'lr': lr
            }, path)

def load_ckpt(path, model, optimizer, device='cuda'):
    if device == 'cpu':
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
#     lr = checkpoint['lr']
    model.eval()
    return model, optimizer, train_losses, val_losses



    
def get_data(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length
    
    t = torch.zeros(batch_size,1).uniform_(0,20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size,1) + t
    
    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:,-output_sequence_length:]

