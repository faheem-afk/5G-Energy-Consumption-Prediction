from settings import lagged_features, static_data, device, batch_size, model_names
from utils import load_model
import shap
import torch
import numpy as np
from datasetMod import EnergyDataset
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

class Feature_Imp():
    def __init__(self):
       
        test_data = pd.read_parquet('data/preprocessed_data/test.parquet')
        test_dataset = EnergyDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
        
        sample = next(iter(test_loader))
        self.seq_batch, self.static_batch, self.ru_batch, self.mo_batch, self.y_batch = sample

        self.flat_input = np.hstack([
        self.seq_batch.view(self.seq_batch.shape[0], -1).numpy(),
        self.static_batch.numpy(),
        self.ru_batch.view(-1, 1).numpy(),
        self.mo_batch.view(-1, 1).numpy()
    ])
        self.background = self.flat_input[:]
        self.values = []
        
        
    def shap_predict(self, input_array, model):
        
        #Split flat input back into separate tensors
        n_features_seq = self.seq_batch.shape[2]  
  
        n_features_static = self.static_batch.shape[1]  

        seq_len = self.seq_batch.shape[1]
        
        #Split input array into parts
        seq_part = input_array[:, :seq_len * n_features_seq].reshape(-1, seq_len, n_features_seq)
        static_part = input_array[:, seq_len * n_features_seq : seq_len * n_features_seq + n_features_static]
        
        ru_part = input_array[:, -2].astype(int)
        mo_part = input_array[:, -1].astype(int)
        
        #Convert to tensors
        seq_tensor = torch.tensor(seq_part, dtype=torch.float32).to(device)
        static_tensor = torch.tensor(static_part, dtype=torch.float32).to(device)
        ru_tensor = torch.tensor(ru_part, dtype=torch.long).to(device)
        mo_tensor = torch.tensor(mo_part, dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(seq_tensor, static_tensor, ru_tensor, mo_tensor).cpu().numpy()
       
        return outputs

    def explainer(self):
        
        features_names = lagged_features + static_data + ['RU_type'] + ['Mode']
        os.makedirs('feature-imp', exist_ok=True)
        
        for ix, i in enumerate(model_names):
            
            model = load_model(f"artifacts/{i}", device)
            fn = lambda arr: self.shap_predict(arr, model)
    
            explain = shap.KernelExplainer(fn, self.background)
            
            shap_value = explain.shap_values(self.flat_input[:30])  
            
            plt.figure()
            
            shap.summary_plot(shap_value, self.flat_input[:30], feature_names=features_names, show=False)
            
            plt.savefig(f"feature-imp/{model_names[ix]}_shap_summary.png", dpi=300, bbox_inches='tight')  
            plt.close()
    