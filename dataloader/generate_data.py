import glob
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import random
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
root_path = "/scratch/jz5952/FoV"
path = "/scratch/jz5952/FoV/dataset"
test_files = glob.glob(path + '/ChenYongting*.txt')
val_files = glob.glob(path + '/fupingyu*.txt') + glob.glob(path + '/GuoYushan*.txt')
train_files = glob.glob(path + '/*.txt')
other_files = glob.glob(path + '/intersection*.txt')+ glob.glob(path + '/original*.txt')+glob.glob(path + '/output*.txt')
train_files = list(filter(lambda i: i not in test_files and (i not in val_files and i not in other_files), train_files))

def get_sine_cosine(roll, pitch, yaw):
    r_sine = np.sin(roll*np.pi/180)
    r_cosine = np.cos(roll*np.pi/180)
    p_sine = np.sin(pitch*np.pi/180)
    p_cosine = np.cos(pitch*np.pi/180)
    y_sine = np.sin(yaw*np.pi/180)
    y_cosine = np.cos(yaw*np.pi/180)
    return pd.Series({'r_sine': r_sine, 'r_cosine': r_cosine, 'p_sine': p_sine, 'p_cosine': p_cosine, 'y_sine': y_sine, 'y_cosine': y_cosine})

# Creates dataframe for the file and resample
def createDataframe(f,denoise_flag=False):
    df = pd.read_csv(f, sep=' ', header=None)
    df.columns = map(lambda x: x.replace(',', ''), df.iloc[0])
    df = df.iloc[1:].astype(float)
    df = df.iloc[:, 0:8]
    sine_cosine_df = df.apply(lambda row: get_sine_cosine(row['HeadRX'], row['HeadRY'], row['HeadRZ']), axis=1)
    df = pd.concat([df.iloc[:, 0:5], sine_cosine_df], axis=1)
    df.index = pd.to_timedelta(df.index, unit='s')
    df = df.resample('200ms').interpolate('akima')  # upsample by 5 --> 5 * 144 Hz
    df = df.resample('2400ms').first()  # downsample by 12 --> (5 * 144) / 12 = 60 Hz
    df = df.reset_index(drop=True)  # drop the timestamp index added
    regex_pattern = '.*Timer|.*Frame'
    filtered_columns = df.filter(regex=regex_pattern, axis=1)
    df = df.drop(columns=filtered_columns.columns).astype(np.float32)
    if denoise_flag:
        df = denoise(df)
    return df

# Creates input and output numpy array for a given dataframe, history_time, target_time and
# step size (in sec)
def multivariate_data(df, history_time = 10, target_time = 10, step = 15, window_size=60):
    data = []
    labels = []
    start_index = int(history_time * window_size)
    end_index = int(len(df) - target_time * window_size)

    for i in range(start_index, end_index, step):
        indices = range(i-int(history_time * window_size), i)
        data.append(df.iloc[indices])
        labels.append(df.iloc[i:i+int(target_time * window_size)])

    return np.array(data), np.array(labels)



def normalizeData(files, history_time = 10, target_time = 10, window_size=60,denoised_flag=False):
    concatenatedDf = pd.DataFrame()
    for f in files:
        df = createDataframe(f,denoise_flag=denoised_flag)
        #print ('f',f, df.columns)
        if len(df) < (history_time + target_time) * (window_size):
            continue
        
        concatenatedDf = pd.concat([concatenatedDf, df], axis=0)
    print ('concatenatedDf.shape, concatenatedDf.columns', concatenatedDf.shape, concatenatedDf.columns)
    HeadX_mean = concatenatedDf['HeadX'].mean()
    HeadY_mean = concatenatedDf['HeadY'].mean()
    HeadZ_mean = concatenatedDf['HeadZ'].mean()
    HeadX_std = concatenatedDf['HeadX'].std()
    HeadY_std = concatenatedDf['HeadY'].std()
    HeadZ_std = concatenatedDf['HeadZ'].std()

    return HeadX_mean, HeadY_mean, HeadZ_mean, HeadX_std, HeadY_std, HeadZ_std


def createSequence(files, history_time=10, target_time=10, step=15, window_size=60,denoise_flag=False, mean_std_train=None):
    x_list = []
    y_list = []
    
    HeadX_mean, HeadY_mean, HeadZ_mean, HeadX_std, HeadY_std, HeadZ_std = normalizeData(files, history_time, target_time, window_size,denoise_flag)
    mean_std = np.array([HeadX_mean, HeadY_mean, HeadZ_mean, HeadX_std, HeadY_std, HeadZ_std])
    if mean_std_train is not None:
        HeadX_mean_train, HeadY_mean_train, HeadZ_mean_train, HeadX_std_train, HeadY_std_train, HeadZ_std_train = mean_std_train
    else:
        HeadX_mean_train, HeadY_mean_train, HeadZ_mean_train, HeadX_std_train, HeadY_std_train, HeadZ_std_train = HeadX_mean, HeadY_mean, HeadZ_mean, HeadX_std, HeadY_std, HeadZ_std
    for f in files:
        df = createDataframe(f,denoise_flag)
        len(df)
        if len(df) < (history_time + target_time) * (window_size):
            continue
        df['HeadX'] = (df['HeadX'] - HeadX_mean_train) / HeadX_std_train
        df['HeadY'] = (df['HeadY'] - HeadY_mean_train) / HeadY_std_train
        df['HeadZ'] = (df['HeadZ'] - HeadZ_mean_train) / HeadZ_std_train

        x_data, y_data = multivariate_data(df, history_time, target_time)
        x_list.append(x_data)
        y_list.append(y_data)

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return x, y, mean_std

def createAndSaveLongSequence(data_path, save_path, history_time = 10, target_time = 10, window_size=60,denoise_flag=False):
    test_files = glob.glob(path + '/ChenYongting*.txt')
    val_files = glob.glob(path + '/fupingyu*.txt') + glob.glob(path + '/GuoYushan*.txt')
    train_files = glob.glob(path + '/*.txt')
    other_files = glob.glob(path + '/intersection*.txt')+ glob.glob(path + '/original*.txt')+glob.glob(path + '/output*.txt')
    train_files = list(filter(lambda i: i not in test_files and (i not in val_files and i not in other_files), train_files))
    HeadX_mean, HeadY_mean, HeadZ_mean, HeadX_std, HeadY_std, HeadZ_std = normalizeData(train_files, history_time, target_time, window_size)
  
    for f in train_files:
        file_name = f.split('/')[-1][:-4]
        # print(file_name)
        df = createDataframe(f,denoise_flag)
        len(df)
        if len(df) < (history_time + target_time) * (window_size):
            continue
        df['HeadX'] = (df['HeadX'] - HeadX_mean) / HeadX_std
        df['HeadY'] = (df['HeadY'] - HeadY_mean) / HeadY_std
        df['HeadZ'] = (df['HeadZ'] - HeadZ_mean) / HeadZ_std
        df.to_csv(f'{save_path}/{file_name}_{history_time}_{target_time}.csv', index=False)
        # print(df)
    return

def generate_data(path, save_path, hist_time, pred_time,frame_rate,denoise_flag=False):
    history_size,target_size = hist_time*frame_rate, pred_time*frame_rate
    test_files = glob.glob(path + '/ChenYongting*.txt')
    val_files = glob.glob(path + '/fupingyu*.txt') + glob.glob(path + '/GuoYushan*.txt')
    train_files = glob.glob(path + '/*.txt')
    other_files = glob.glob(path + '/intersection*.txt')+ glob.glob(path + '/original*.txt')+glob.glob(path + '/output*.txt')
    train_files = list(filter(lambda i: i not in test_files and (i not in val_files and i not in other_files), train_files))
    x_train,y_train, mean_std_train = createSequence(train_files,hist_time,pred_time,window_size=frame_rate,denoise_flag=denoise_flag, mean_std_train=None)
    x_val,y_val, mean_std_val = createSequence(val_files,hist_time,pred_time,window_size=frame_rate,denoise_flag=denoise_flag,mean_std_train=mean_std_train)
    x_test,y_test, mean_std_test = createSequence(test_files,hist_time,pred_time,window_size=frame_rate,denoise_flag=denoise_flag,mean_std_train=mean_std_train)
    mean_std = np.concatenate((mean_std_train, mean_std_val,mean_std_test))
    # save to processed data
    np.savetxt(f'{save_path}/x_train_{hist_time}_{pred_time}.csv', x_train.reshape((x_train.shape[0],-1)), delimiter=',')
    np.savetxt(f'{save_path}/y_train_{hist_time}_{pred_time}.csv', y_train.reshape((y_train.shape[0],-1)), delimiter=',')
    np.savetxt(f'{save_path}/x_val_{hist_time}_{pred_time}.csv', x_val.reshape((x_val.shape[0],-1)), delimiter=',')
    np.savetxt(f'{save_path}/y_val_{hist_time}_{pred_time}.csv', y_val.reshape((y_val.shape[0],-1)), delimiter=',')
    np.savetxt(f'{save_path}/x_test_{hist_time}_{pred_time}.csv', x_test.reshape((x_test.shape[0],-1)), delimiter=',')
    np.savetxt(f'{save_path}/y_test_{hist_time}_{pred_time}.csv', y_test.reshape((y_test.shape[0],-1)), delimiter=',')
    np.savetxt(f'{save_path}/xyz_mean_std_{hist_time}_{pred_time}.csv',mean_std)
    return x_train, y_train, x_val,y_val,x_test,y_test,mean_std

def denoise(df, window_length=30, polyorder=2, deriv=0, delta=1.0, axis=-2, mode='interp'):
    """Denoise the data using Savitzky-Golay filter.
    Args:
        data (dataframe): The data to denoise.
        window_length (int): The length of the filter window.
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        dataframe: The denoised data.
    """
    data = df.to_numpy()
    denoised_data = savgol_filter(data, window_length, polyorder,deriv, delta, axis, mode)
    # plt.figure()
    # fig, ax = plt.subplots(1, 3, figsize=(12,4))
    # rand_idx = 0
    # if len(data) > 400:
    #     rand_idx = random.randint(0,len(data)-400)
    # end_idx = min(rand_idx+400,len(data))
    # ax[0].plot(data[rand_idx:end_idx,0])
    # ax[1].plot(data[rand_idx:end_idx,1])
    # ax[2].plot(data[rand_idx:end_idx,2])
    # denoised_data = savgol_filter(data, window_length, polyorder, deriv=deriv, delta=delta, axis=axis, mode=mode)
    # ax[0].plot(denoised_data[rand_idx:end_idx,0])
    # ax[1].plot(denoised_data[rand_idx:end_idx,1])
    # ax[2].plot(denoised_data[rand_idx:end_idx,2])
    # plt.legend(["original","denoised"])
    # # plt.show()
    # fig.savefig(f'/scratch/jz5952/FoV/denoised_examples/denoised_example_{rand_idx}.png')
    return pd.DataFrame(denoised_data, columns=df.columns)


def generate_train_data(path, hist_time, pred_time,frame_rate):
    history_size,target_size = hist_time*frame_rate, pred_time*frame_rate
    test_files = glob.glob(path + '/ChenYongting*.txt')
    val_files = glob.glob(path + '/fupingyu*.txt') + glob.glob(path + '/GuoYushan*.txt')
    train_files = glob.glob(path + '/*.txt')
    other_files = glob.glob(path + '/intersection*.txt')+ glob.glob(path + '/original*.txt')+glob.glob(path + '/output*.txt')
    train_files = list(filter(lambda i: i not in test_files and (i not in val_files and i not in other_files), train_files))
    

    
    
    
    