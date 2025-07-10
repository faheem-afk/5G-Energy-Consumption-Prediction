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
        # read test data from parquet and make dataset
        test_data = pd.read_parquet('data/preprocessed_data/test.parquet')
        test_dataset = EnergyDataset(test_data)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        sample = next(iter(test_loader))
        self.seq_batch, self.static_batch, self.ru_batch, self.mo_batch, self.y_batch = sample

        # flatten sequence and static and embeddings into single input array
        self.flat_input = np.hstack([
            self.seq_batch.view(self.seq_batch.shape[0], -1).numpy(),
            self.static_batch.numpy(),
            self.ru_batch.view(-1, 1).numpy(),
            self.mo_batch.view(-1, 1).numpy()
        ])
   
        self.background = self.flat_input[:]
        self.values = []  # list to keep SHAP values later
        
    def shap_predict(self, input_array, model):
    
        n_features_seq = self.seq_batch.shape[2]
        n_features_static = self.static_batch.shape[1]
        seq_len = self.seq_batch.shape[1]
        
        # split flat array back into seq, static, ru, mo parts
        seq_part = input_array[:, :seq_len * n_features_seq]\
                   .reshape(-1, seq_len, n_features_seq)
        static_part = input_array[:, seq_len * n_features_seq:
                                   seq_len * n_features_seq + n_features_static]
        ru_part = input_array[:, -2].astype(int)  # RU type index
        mo_part = input_array[:, -1].astype(int)  # Mode index
        
        seq_tensor = torch.tensor(seq_part, dtype=torch.float32).to(device)
        static_tensor = torch.tensor(static_part, dtype=torch.float32).to(device)
        ru_tensor = torch.tensor(ru_part, dtype=torch.long).to(device)
        mo_tensor = torch.tensor(mo_part, dtype=torch.long).to(device)
        
        # do prediction without grad
        with torch.no_grad():
            outputs = model(seq_tensor, static_tensor, ru_tensor, mo_tensor).cpu().numpy()
       
        return outputs  # return numpy predictions

    def explainer(self):
     
        features_names = lagged_features + static_data + ['RU_type'] + ['Mode']
     
        os.makedirs('feature-imp', exist_ok=True)
        
        # for each model name, load and explain
        for ix, name in enumerate(model_names):
            model = load_model(f"artifacts/{name}", device)  # load saved model
            # define prediction function for SHAP
            fn = lambda arr: self.shap_predict(arr, model)
    
            explain = shap.KernelExplainer(fn, self.background)  # make explainer
            
            # compute SHAP values for first 30 samples
            shap_value = explain.shap_values(self.flat_input[:30])  
            
            plt.figure()  # new figure
            # make summary plot but do not show
            shap.summary_plot(shap_value, self.flat_input[:30],
                              feature_names=features_names, show=False)
            # save plot to file with high resolution
            plt.savefig(f"feature-imp/{name}_shap_summary.png",
                        dpi=300, bbox_inches='tight')
            plt.close()  # close figure to free memory