import torch
from utils import read_from_yaml
from data_preprocessing import DataPreprocessing

#Setting the device value based on the architecture of the system 
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps"   if torch.has_mps else "cpu")

#When settings module is used to import lagged_features, they get be imported in the reverse order
lagged_features, static_data, _ = read_from_yaml('variables.yaml')
lagged_features.reverse()

#Saving the RUType and Mode Categories
dataPreprocessingObject = DataPreprocessing()
dataPreprocessingObject.pre_processing()

ru_num_embeddings = len(dataPreprocessingObject.num_ru_categories)
mo_num_embeddings = len(dataPreprocessingObject.num_mo_categories)


batch_size = 16
n_epochs = 55
lr= 0.002200306080968846
weight_decay= 0.0004055893568937846
factor= 0.30000000000000004
min_lr=1e-9
patience=3
cnn_in_channels  = 4
cnn_out_channels1 = 32
cnn_kernel_size1  = 3
hidden_size = 32
num_layers  = 1
ru_embedding_dim = 8
mo_embedding_dim = 8
static_feature_dim= len(static_data)
fc_hidden_dim     = 192
dropout_prob      = 0.1
n_splits=10
mode='min'
model_names = ['GRU', 'LSTM', 'RNN']