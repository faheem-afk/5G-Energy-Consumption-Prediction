# import necessary libraries
import torch.nn as nn
import torch
import numpy as np  
from settings import static_data, device
from datasetMod import EnergyDataset  
from sklearn.model_selection import GroupKFold  
from torch.utils.data import DataLoader  
from data_preprocessing import DataPreprocessing  
from Models import CNN_RNN_EnergyModel  
from torch import optim  
import optuna  
from optuna.samplers import TPESampler  

# define objective function for Optuna trials
def objective(trial):
    # sample hyperparameters with TPE sampler
    cnn_out1     = trial.suggest_int("cnn_out1",  8, 32, step=4)
    kernel1      = trial.suggest_int("kernel1",   1, 6)
    model_hidden_dim = trial.suggest_int("hidden_dim",  32, 256, step=32)
    layers       = trial.suggest_int("layers",    1, 3)
    ru_embed_dim = trial.suggest_int("ru_embed_dim",  8, 32, step=8)
    mo_embed_dim = trial.suggest_int("mo_embed_dim",  8, 32, step=8)
    fc_dim       = trial.suggest_int("fc_dim",       32, 256, step=32)
    dropout_p    = trial.suggest_float("dropout",   0.1, 0.5, step=0.1)
    lr           = trial.suggest_float("lr",      1e-4, 1e-2, log=True)
    wd           = trial.suggest_float("wd",     1e-5, 1e-3, log=True)
    batch_size   = trial.suggest_int("batch_size",  8, 64, step=8)
    patience     = trial.suggest_int('patience', 2, 5, step=1)
    factor       = trial.suggest_float('factor', 0.2, 0.6, step=0.1)

    # load and preprocess data
    dataPreprocessingObject = DataPreprocessing()
    train_data, _ = dataPreprocessingObject.pre_processing()

    # use GroupKFold to avoid data leak by base station groups
    gkf    = GroupKFold(n_splits=5)
    groups = train_data["BS"].values

    # list to store each fold best val loss
    fold_val_losses = []
    fold = 1

    # loop over each fold split
    for train_idx, valid_idx in gkf.split(train_data, train_data["Energy"], groups=groups):

        # create model instance with sampled hyperparams
        model = CNN_RNN_EnergyModel(
            cnn_in_channels     = 4,
            cnn_out_channels1   = cnn_out1,
            cnn_kernel_size1    = kernel1,
            rnn_hidden_size     = model_hidden_dim,
            rnn_num_layers      = layers,
            ru_num_embeddings   = train_data["RUType_enc"].nunique(),
            ru_embedding_dim    = ru_embed_dim,
            mo_num_embeddings   = train_data["Mode_enc"].nunique(),
            mo_embedding_dim    = mo_embed_dim,
            static_feature_dim  = len(static_data),
            fc_hidden_dim       = fc_dim,
            dropout_prob        = dropout_p
        )

        # define loss function and optimizer
        criterion = nn.L1Loss()  # use MAE
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        # scheduler reduce lr when val loss not improve
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=1e-15,
        )

        # prepare data loaders
        train_df = train_data.loc[train_idx]
        validation_df = train_data.loc[valid_idx]
        train_dataset = EnergyDataset(train_df)
        val_dataset = EnergyDataset(validation_df)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # training parameters
        n_epochs = 50
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            train_loss = 0.0
            model.train() 
            for x_seq_b, x_stat_b, ru_b, mo_b, y_b in train_loader:
             
                x_seq_b, x_stat_b, ru_b, mo_b, y_b = (
                    x_seq_b.to(device), x_stat_b.to(device), ru_b.to(device),
                    mo_b.to(device),  y_b.to(device).squeeze(-1)
                )
                optimizer.zero_grad()
                y_hat = model(x_seq_b, x_stat_b, ru_b, mo_b)
                loss = criterion(y_hat, y_b)
                train_loss += loss.item() * x_seq_b.size(0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            avg_train_loss = train_loss / len(train_dataset)

            # validation phase
            model.eval()
            valid_loss_acc = 0.0
            with torch.no_grad():
                for x_seq_b, x_stat_b, ru_b, mo_b, y_b in val_loader:
                    x_seq_b, x_stat_b, ru_b, mo_b, y_b = (
                        x_seq_b.to(device), x_stat_b.to(device), ru_b.to(device),
                        mo_b.to(device), y_b.to(device).squeeze(-1)
                    )
                    y_hat = model(x_seq_b, x_stat_b, ru_b, mo_b)
                    val_loss = criterion(y_hat, y_b)
                    valid_loss_acc += val_loss.item() * x_seq_b.size(0)

            avg_val_loss = valid_loss_acc / len(val_dataset)
            scheduler.step(avg_val_loss)

            # print training and val metrics
            print(
                f"Epoch {epoch+1}/{n_epochs}  "
                f"Train MAE: {avg_train_loss:.4f}  "
                f" Val MAE: {avg_val_loss:.4f}  "
                f" LR={optimizer.param_groups[0]['lr']:.2e}"
            )

            # report to Optuna and maybe prune
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # update best val loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        # after fold, store best loss
        fold_val_losses.append(best_val_loss)
        print(f"======================== Fold {fold} is Completed ========================")
        fold += 1

        torch.cuda.empty_cache()

    final_val_loss = np.mean(fold_val_losses)
    print(f"===========================Avg val loss across the folds={final_val_loss}===========================")
    return final_val_loss


if __name__ == "__main__":

    tpe_sampler = TPESampler(n_startup_trials=60)

    # create study to minimize objective
    study = optuna.create_study(sampler=tpe_sampler, direction="minimize")

    # run optimization trials
    study.optimize(objective, n_trials=2000, timeout=720000000)

    # show result of best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation loss: {trial.value:.5f}")
    print("  Hyperparameters:")
    for key, val in trial.params.items():
        print(f"    {key}: {val}")