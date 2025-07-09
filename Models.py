import torch
import torch
import torch.nn as nn
from settings import *
import warnings

warnings.filterwarnings("ignore")


class CNN_RNN_EnergyModel(nn.Module):
    def __init__(self,
                ru_num_embeddings: int = ru_num_embeddings,
                mo_num_embeddings: int = mo_num_embeddings,
                cnn_in_channels: int = cnn_in_channels,
                cnn_out_channels1: int = cnn_out_channels1,
                cnn_kernel_size1: int   = cnn_kernel_size1,
                hidden_size: int    = hidden_size,
                num_layers: int     = num_layers,
                ru_embedding_dim: int   = ru_embedding_dim,
                mo_embedding_dim: int   = mo_embedding_dim,
                static_feature_dim: int = len(static_data),
                fc_hidden_dim: int      = fc_hidden_dim,
                dropout_prob: float     = dropout_prob,
                 ):
        
        super().__init__()
        
        #1D‐CNN
        self.cnn1 = nn.Conv1d(
            in_channels  = cnn_in_channels,
            out_channels = cnn_out_channels1,
            kernel_size  = cnn_kernel_size1,
            stride        = 1,
            padding       = 0
        )
        self.act1 = nn.SiLU()
        
        #simple RNN
        self.rnn = nn.RNN(
            input_size   = cnn_out_channels1,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout_prob,
            bidirectional= False
        )
        
        #embeddings for RUType and Mode
        self.ru_emb = nn.Embedding(ru_num_embeddings, ru_embedding_dim)
        self.mo_emb = nn.Embedding(mo_num_embeddings, mo_embedding_dim)
        
        #final MLP head
        combined_dim = (
            hidden_size
            + ru_embedding_dim
            + mo_embedding_dim
            + static_feature_dim
        )
        self.fc1 = nn.Linear(combined_dim, fc_hidden_dim)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        
    def forward(self, x_seq, x_static, ru_idx, mo_idx):
        
        #CNN expects (batch, channels, seq_len)
        x = x_seq.permute(0, 2, 1)    
        x = self.cnn1(x)            
        x = self.act1(x)
        
        #back to (batch, new_len, channels)
        x = x.permute(0, 2, 1)    
        
        _, h_n = self.rnn(x)
        
        #last layer’s hidden state
        seq_feat = h_n[-1]           
        
        # embeddings
        ru_emb = self.ru_emb(ru_idx) 
        mo_emb = self.mo_emb(mo_idx) 
        
        # concat
        combined = torch.cat([seq_feat, ru_emb, mo_emb, x_static], dim=1)
        
        # head
        h = self.fc1(combined)
        h = self.act2(h)
        h = self.drop(h)
        out = self.fc2(h)
        
        return out.squeeze(1)
    

class CNN_LSTM_EnergyModel(nn.Module):
    def __init__(self,
                ru_num_embeddings: int = ru_num_embeddings,
                mo_num_embeddings: int = mo_num_embeddings,
                cnn_in_channels: int = cnn_in_channels,
                cnn_out_channels1: int = cnn_out_channels1,
                cnn_kernel_size1: int   = cnn_kernel_size1,
                hidden_size: int    = hidden_size,
                num_layers: int     = num_layers,
                ru_embedding_dim: int   = ru_embedding_dim,
                mo_embedding_dim: int   = mo_embedding_dim,
                static_feature_dim: int = len(static_data),
                fc_hidden_dim: int      = fc_hidden_dim,
                dropout_prob: float     = dropout_prob,
                 ):
        
        super().__init__()
        
        #1D‐CNN
        self.cnn1 = nn.Conv1d(
            in_channels  = cnn_in_channels,
            out_channels = cnn_out_channels1,
            kernel_size  = cnn_kernel_size1,
            stride        = 1,
            padding       = 0
        )
        self.act1 = nn.SiLU()
        
        #simple lstm
        self.lstm = nn.LSTM(
            input_size   = cnn_out_channels1,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout_prob,
            bidirectional= False
        )
        
        #embeddings for RUType and Mode
        self.ru_emb = nn.Embedding(ru_num_embeddings, ru_embedding_dim)
        self.mo_emb = nn.Embedding(mo_num_embeddings, mo_embedding_dim)
        
        #final MLP head
        combined_dim = (
            hidden_size
            + ru_embedding_dim
            + mo_embedding_dim
            + static_feature_dim
        )
        self.fc1 = nn.Linear(combined_dim, fc_hidden_dim)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        
    def forward(self, x_seq, x_static, ru_idx, mo_idx):
   
        #CNN expects (batch, channels, seq_len)
        x = x_seq.permute(0, 2, 1)    
        x = self.cnn1(x)             
        x = self.act1(x)
        
        #back to (batch, new_len, channels)
        x = x.permute(0, 2, 1)     
        

        _, (h_n, _) = self.lstm(x)

        # last layer’s hidden state
        seq_feat = h_n[-1]          
        
        # embeddings
        ru_emb = self.ru_emb(ru_idx)
        mo_emb = self.mo_emb(mo_idx)
        
        # concat
        combined = torch.cat([seq_feat, ru_emb, mo_emb, x_static], dim=1)
        
        # head
        h = self.fc1(combined)
        h = self.act2(h)
        h = self.drop(h)
        out = self.fc2(h)
        
        return out.squeeze(1)
    

class CNN_GRU_EnergyModel(nn.Module):
    def __init__(self,
                ru_num_embeddings: int = ru_num_embeddings,
                mo_num_embeddings: int = mo_num_embeddings,
                cnn_in_channels: int = cnn_in_channels,
                cnn_out_channels1: int = cnn_out_channels1,
                cnn_kernel_size1: int   = cnn_kernel_size1,
                hidden_size: int    = hidden_size,
                num_layers: int     = num_layers,
                ru_embedding_dim: int   = ru_embedding_dim,
                mo_embedding_dim: int   = mo_embedding_dim,
                static_feature_dim: int = len(static_data),
                fc_hidden_dim: int      = fc_hidden_dim,
                dropout_prob: float     = dropout_prob,
                 ):
        
        super().__init__()
        
        #1D‐CNN
        self.cnn1 = nn.Conv1d(
            in_channels  = cnn_in_channels,
            out_channels = cnn_out_channels1,
            kernel_size  = cnn_kernel_size1,
            stride        = 1,
            padding       = 0
        )
        self.act1 = nn.SiLU()
        
        #simple gru
        self.gru = nn.GRU(
            input_size   = cnn_out_channels1,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout_prob,
            bidirectional= False
        )
        
        #embeddings for RUType and Mode
        self.ru_emb = nn.Embedding(ru_num_embeddings, ru_embedding_dim)
        self.mo_emb = nn.Embedding(mo_num_embeddings, mo_embedding_dim)
        
        #final MLP head
        combined_dim = (
            hidden_size
            + ru_embedding_dim
            + mo_embedding_dim
            + static_feature_dim
        )
        self.fc1 = nn.Linear(combined_dim, fc_hidden_dim)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        
    def forward(self, x_seq, x_static, ru_idx, mo_idx):
   
        #CNN expects (batch, channels, seq_len)
        x = x_seq.permute(0, 2, 1)    
        x = self.cnn1(x)             
        x = self.act1(x)
        
        #back to (batch, new_len, channels)
        x = x.permute(0, 2, 1)       
        
        _, h_n = self.gru(x)
        
        # last layer’s hidden state
        seq_feat = h_n[-1]           
        
        # embeddings
        ru_emb = self.ru_emb(ru_idx) 
        mo_emb = self.mo_emb(mo_idx) 
        
        # concat
        combined = torch.cat([seq_feat, ru_emb, mo_emb, x_static], dim=1)
        
        # head
        h = self.fc1(combined)
        h = self.act2(h)
        h = self.drop(h)
        out = self.fc2(h)
        
        return out.squeeze(1)
