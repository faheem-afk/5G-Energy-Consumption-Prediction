from torch.utils.data import Dataset
from utils import read_from_yaml
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class EnergyDataset(Dataset):
    def __init__(self, df):
       
        super().__init__()
        self.df = df
        
        lagged_features, static_data, target_col = read_from_yaml('variables.yaml')
        
        # We need the lagged features in the reverse order for the model to learn in the sequential manner, i.e.., T3, T2, T1
        lagged_features.reverse()
        
        self.x_seq = (
            df[lagged_features]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 3, 4)
        )
        
        # These will be the static features in the data
        self.x_static = df[static_data].to_numpy(dtype=np.float32)
        
        # These will be the categorical features
        self.x_ru = df['RUType_enc'].to_numpy(dtype=np.int64)

        self.x_mo = df['Mode_enc'].to_numpy(dtype=np.int64)
        
        # Target value
        self.y = df[target_col].to_numpy(dtype=np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
 
        return (
            self.x_seq[idx],          
            self.x_static[idx],      
            self.x_ru[idx],
            self.x_mo[idx],           
            self.y[idx]
        )
