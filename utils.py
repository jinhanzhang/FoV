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
feature_size = FEATURE_SIZE
print(feature_size)
x_data = x_val[::8,:,:].reshape(-1,9)
print(x_data.shape)
y_data = y_val[::8,:,:].reshape(-1,9)
plt.figure()
if feature_size<=3:
    fig, ax = plt.subplots(1, feature_size, figsize=(feature_size*4,4))
    for i in range(feature_size):
        ax[i].plot(x_data[:,i])
        ax[i].plot(y_data[:,i])
else:
    rows = (feature_size-1)//3+1
    fig, ax = plt.subplots(rows, 3, figsize=(12,4*rows))
    for i in range(rows):
        for j in range(3):
            if i*3+j<feature_size:
                ax[i][j].plot(x_data[:,3*i+j])
                ax[i][j].plot(y_data[:,3*i+j])
fig.legend(["data","target"])
plt.show()




    
def get_data(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length
    
    t = torch.zeros(batch_size,1).uniform_(0,20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size,1) + t
    
    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:,-output_sequence_length:]

