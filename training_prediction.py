import json
import torch.nn as nn
import torch
import numpy as np
import Models
from settings import *
from datasetMod import EnergyDataset
import inspect
from utils import seed_everything, set_seed, save_model, load_model
import time
from logger import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader
import warnings
import pandas as pd
import os
import re


warnings.filterwarnings("ignore")


class TrainingPrediction():
   
    def __init__(self, dataPreprocessingObject):
       self.dataPreprocessingObject = dataPreprocessingObject
  
    def training_and_validation(self):
        
        # Setting the seed value for reproducibility
        set_seed(42)
        seed_everything(42)
        
        train_data, test_data =  self.dataPreprocessingObject.pre_processing()
        
        # Using GroupKfold for training, so we would have different base_stations while training and different ones while validation
        gkf = GroupKFold(n_splits=n_splits)
        groups = train_data['BS']

        # Get all classes defined in that module
        classes = inspect.getmembers(Models, inspect.isclass)

        # Filter to include only classes defined in that module
        module_classes = [cls for name, cls in classes if cls.__module__ == Models.__name__]
        
        # Filter to include only names defined in that module
        module_names = [name for name, cls in classes if cls.__module__ == Models.__name__]
        
        pattern = re.compile(r"rnn|gru|lstm", flags=re.IGNORECASE)


        # Looping through the models 
        for cls_idx, cls in enumerate(module_classes):
            all_scores = []
            all_errors = []

            match = re.search(pattern, module_names[cls_idx])

            # Looping through the folds for each model
            fold = 1
            for train_idx, valid_idx in gkf.split(train_data, train_data['Energy'], groups=groups):
                
                # Creating an instance of the model for each fold
                model = cls().to(device)
                
                parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print("\n")
                
                logging.info(f"Fold {fold}/{n_splits} ..\n")
            
                logging.info(f"{match.group()} instantiated with {parameters} params ..\n")
                
                criterion = nn.L1Loss()
                optimizer = torch.optim.AdamW(model.parameters(), 
                                            lr = lr, 
                                            weight_decay = weight_decay)
                
                # Implements early stopping, so as to prevent overfitting
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=mode,
                    factor= factor,
                    patience=patience,
                    min_lr=min_lr,    
                )
                
                # Now, using the index values provided by groupKfold, we will create our train, and validation sets
                train_df = train_data.loc[train_idx]
                validation_df = train_data.loc[valid_idx]
            
                train_dataset = EnergyDataset(train_df)
                val_dataset = EnergyDataset(validation_df)
                
                # For prediction purpose after each fold
                test_dataset = EnergyDataset(test_data)
                
                train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
                
                # For prediction purpose after each fold
                test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
                
                # This will store the best value found for validation loss throughout the epochs in each fold
                best_val_loss = float('inf')
                epochs_no_improve = 0

                # Recording avg time for an epoch in fold 1 only
                if fold == 1:
                    start = time.time()
                
                logging.info(f"Training started on {match.group()} ..\n")
                for epoch in range(n_epochs):
                    model.train()
                    train_loss_accum = 0.0
                    
                    for seq_batch, static_batch, ru_batch, mo_batch, y_batch in train_loader:
                    
                        seq_batch   = seq_batch.to(device)        
                        static_batch= static_batch.to(device)     
                        ru_batch    = ru_batch.to(device)         
                        mo_batch    = mo_batch.to(device)
                        y_batch     = y_batch.to(device).squeeze(-1)        
                        
                        optimizer.zero_grad()
                        y_pred = model(seq_batch, static_batch, ru_batch, mo_batch)
                        
                        loss = criterion(y_pred, y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # The loss value for a batch will be used as an avg loss in that batch, therefore, the loss will be mutliplied by the number of items in a batch, so as to provide a loss value for each sample
                        train_loss_accum += loss.item() * seq_batch.size(0)
                    
                    # Now, since we have a loss value for each sample, we could calculate an avg loss per sample 
                    avg_train_loss = train_loss_accum / len(train_dataset)
                    
                    model.eval()
                    val_loss_accum = 0.0
                    with torch.no_grad():
                        for seq_batch, static_batch, ru_batch, mo_batch, y_batch in val_loader:
                            seq_batch    = seq_batch.to(device)
                            static_batch = static_batch.to(device)
                            ru_batch     = ru_batch.to(device)
                            mo_batch     = mo_batch.to(device)
                            y_batch      = y_batch.to(device).squeeze(-1)
                            
                            y_pred = model(seq_batch, static_batch, ru_batch, mo_batch)
                            val_loss = criterion(y_pred, y_batch)
                            val_loss_accum += val_loss.item() * seq_batch.size(0)
                    
                    avg_val_loss = val_loss_accum / len(val_dataset)
                    scheduler.step(avg_val_loss)
        
                    logging.info(
                    f"Epoch {epoch+1}/{n_epochs}  "
                    f"Train MAE: {avg_train_loss:.4f}  "
                    f" Val MAE: {avg_val_loss:.4f}  "
                    f" LR={optimizer.param_groups[0]['lr']:.2e}"
                    f"\n"
                    )
                    
                    if avg_val_loss < best_val_loss - 1e-4:
                            best_val_loss = avg_val_loss
                            epochs_no_improve = 0
                    else:
                            epochs_no_improve += 1
                            logging.info(f"===========================epochs_no_improve: {epochs_no_improve}===========================\n")
                    if epochs_no_improve >= 8:
                        break
                
                logging.info(f"Training finished ..\n")
                if fold == 1:
                    end = time.time()
                    
                    diff = round((end - start), 2)
                        
                    logging.info(f"Avg epoch time {str(diff / n_epochs)}s\n")
                
                logging.info(f"Saving the model ..\n")    
                save_model(model, f"artifacts/{match.group()}")
                logging.info(f"Saving Complete ..\n")

                logging.info(f"Loading the model ..\n")
                model = load_model(f"artifacts/{match.group()}", device)
                logging.info(f"Model loaded ..\n")
                
                # Evaluating the model on the test set
                logging.info(f"Running Evaluation on the test set ..\n")
                
                model.eval()
                all_preds = []
                all_trues = []
                with torch.no_grad():
                    for seq_batch, static_batch, ru_batch, mo_batch, y_batch in test_loader:
                        seq_batch    = seq_batch.to(device)
                        static_batch = static_batch.to(device)
                        ru_batch     = ru_batch.to(device)
                        mo_batch     = mo_batch.to(device)
                        y_batch      = y_batch.to(device).squeeze(-1)
                
                        y_hat = model(seq_batch, static_batch, ru_batch, mo_batch)
                    
                        preds_np = y_hat.cpu().numpy()    
                        true_np  = y_batch.cpu().numpy().ravel()    
                        all_preds.append(preds_np)
                        all_trues.append(true_np)

            
                all_preds = np.concatenate(all_preds)  
                all_trues = np.concatenate(all_trues)
        
                fold_mae  = mean_absolute_error(all_trues, all_preds)
                fold_mape = mean_absolute_percentage_error(all_trues, all_preds)

                logging.info(f"Final MAE on test data for fold {fold}: {fold_mae:.4f}, MAPE: {fold_mape:.4f}\n")
                all_scores.append(fold_mae)
                all_errors.append(fold_mape)
                
                logging.info(f"Evaluation complete ..\n")
                fold+=1

            logging.info(f"MAE across folds : {all_scores}\n")
            logging.info(f"Avg MAE across folds: {np.mean(all_scores)}\n")
            
            os.makedirs('results', exist_ok=True)
            pd.Series(all_scores, name="MAE").to_csv(f"results/{match.group()}.csv", index=False)
            
            df_mae = pd.read_csv(f"results/{match.group()}.csv")
            df_mae['MAPE'] = all_errors
            df_mae.to_csv(f"results/{match.group()}.csv", index=False)
            
            # Storing avg time per epoch information
            os.makedirs('Training-specs', exist_ok=True)
                
            time_dic = {
                "Avg epoch time": f"{(diff / n_epochs)}s",
                "Total Time taken": f"{(diff / n_epochs) * n_splits}s",
                "Training-device":f"{device}",
                "Parameters":f"{parameters}"
            }    
            with open(f'Training-specs/{match.group()}.json', 'w') as f:
                json.dump(time_dic, f, indent=2)
            
          