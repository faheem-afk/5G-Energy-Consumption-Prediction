import pandas as pd
import numpy as np
import os
import pickle
import random
from tqdm import tqdm
from sklearn.preprocessing import SplineTransformer
from scipy.signal import savgol_filter as sg
from scipy.signal import sosfiltfilt, butter, sosfilt, sosfilt_zi
import torch
import warnings

warnings.filterwarnings("ignore")


def load_data():
    df_total = pd.read_parquet("data/total.parquet", engine='pyarrow')
    df_bs = pd.read_parquet("data/bs.parquet", engine='pyarrow')
    df_cell = pd.read_parquet("data/cell.parquet", engine='pyarrow')
    
    return df_total, df_bs, df_cell
        
def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def add_sg(df):
    w = 5  # Window length
    p = 3  # Polynomial order

    for si in tqdm(df.BS.unique()):
        index = df.BS == si

        # Ensure window size does not exceed the size of the data
        group_size = len(df.loc[index, 'load'])
        if group_size >= w:
            df.loc[index, 'load_smooth'] = sg(df.loc[index, 'load'], w, p)
            df.loc[index, 'load_diff'] = sg(df.loc[index, 'load'], w, p, 1)
            df.loc[index, 'load_diff2'] = sg(df.loc[index, 'load'], w, p, 2)
            df.loc[index, 'load_diff3'] = sg(df.loc[index, 'load'], w, p, 3)
        else:
            print(f"Skipping BS {si} because its data length ({group_size}) is smaller than window length {w}.")



# SOS Filter function
def add_sosfiltfilt(df):
    for si in tqdm(df.BS.unique()):
        index = df.BS == si

        #Check the length of the data for this group
        group_size = len(df.loc[index, 'load'])
        
        #Skip groups with insufficient data for filtering
        if group_size > 15 :
            # Define the filter coefficients
            sos = butter(4, 0.125, output='sos')  # 4th order low-pass filter
            sos8 = butter(8, 0.125, output='sos')  # 8th order low-pass filter

            #Apply sosfiltfilt (which handles initial conditions automatically)
            df.loc[index, 'load_sosfiltfilt'] = sosfiltfilt(sos, df.loc[index, 'load'])

            #Apply sosfilt with initial condition using sos8
            zi = np.array(df.loc[index, 'load'][:4]).mean() * sosfilt_zi(sos8)  
            # Calculate initial condition, to help the filter not go nuts in the beginning
            df.loc[index, 'load_sosfilt'], _ = sosfilt(sos8, df.loc[index, 'load'], zi=zi)
        else:
            print(f"Skipping BS {si} because its data length ({group_size}) is smaller than required for filtering.")


import yaml

def read_from_yaml(file_path):
    with open(file_path, "r") as file:
        column_config = yaml.safe_load(file)

    lagged_features = column_config["lagged_features"]
    static_data = column_config["static_data"]
    target_col = column_config["target_col"]
    
    return lagged_features, static_data, target_col


def save_model(model, filepath):

    dir = os.path.dirname(filepath)
    os.makedirs(dir, exist_ok=True)
    
    model_cpu = model.to("cpu")
    
    with open(filepath, "wb") as f:
        pickle.dump(model_cpu, f)
        

def load_model(filepath, device):
    
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    model = model.to(device)
    
    return model
    
    
