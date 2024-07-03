# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DICE model from https://github.com/UsmanMahmood27/DICE """
from random import uniform, randint

import torch
from torch import nn
from torch import optim

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return MultivariateTSModel(model_cfg)


def get_criterion(cfg: DictConfig, model_cfg: DictConfig):
    return RegCEloss(model_cfg)


class RegCEloss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg, lambda1 = 0.01):
        self.ce_loss = nn.CrossEntropyLoss()

        self.reg_param = model_cfg.reg_param

        self.lambda1 = lambda1 # Sparsity loss weight

    def __call__(self, logits, target, model, device, DNC, DNCs):
        ce_loss = self.ce_loss(logits, target)

        # DAG-ness loss - a differentiabe measure of how distant the given DNC is from DAG space
        E = torch.linalg.matrix_exp(DNC * DNC) # (Zheng et al. 2018)
        dag_loss = torch.mean(torch.vmap(torch.trace)(E)) - DNC.shape[1]

        # Sparsity loss on DNC
        sparse_loss = self.lambda1 * torch.mean(torch.sum(torch.abs(DNC), dim=(1,2)))
        # sparse_loss = 0 

        # loss = ce_loss + dag_loss + sparse_loss
        loss = ce_loss
        return loss


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "rnn": {
            "num_layers": 1,
            "input_embedding_size": 32,
            "hidden_embedding_size": 64,
        },
        "reg_param": 1e-6,
        "lr": 2e-4,
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        "rnn": {
            "num_layers": randint(1, 3),
            "input_embedding_size": randint(16, 128),
            "hidden_embedding_size": randint(32, 256),
        },
        "reg_param": 10 ** uniform(-8, -4),
        "lr": 10 ** uniform(-5, -3),
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)

import torch
import torch.nn as nn

class MultivariateTSModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(MultivariateTSModel, self).__init__()

        self.num_components = num_components = model_cfg.input_size
        self.num_layers = num_layers = model_cfg.rnn.num_layers
        self.embedding_dim = embedding_dim = model_cfg.rnn.input_embedding_size
        self.hidden_dim = hidden_dim = model_cfg.rnn.hidden_embedding_size
        output_size = model_cfg.output_size
        
        # Component-specific embeddings
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_components)
        ])
        
        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

        # Global Temporal Attention 
        self.upscale = 0.05
        self.upscale2 = 0.5

        self.HW = torch.nn.Hardswish()
        self.gta_embed = nn.Sequential(
            nn.Linear(
                num_components**2,
                round(self.upscale * num_components**2),
            ),
        )
        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * num_components**2)),
            nn.ReLU(),
        )
        self.gta_attend = nn.Sequential(
            nn.Linear(
                round(self.upscale * num_components**2),
                round(self.upscale2 * num_components**2),
            ),
            nn.ReLU(),
            nn.Linear(round(self.upscale2 * num_components**2), 1),
        )

        # Classifier
        self.clf = nn.Linear(num_components**2, output_size)

    def gta_attention(self, x, node_axis=1):
        # x.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        x_readout = x.mean(node_axis, keepdim=True)
        x_readout = x * x_readout

        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1, x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
        x_graphattention = self.HW(x_graphattention.reshape(a, b))
        return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)
    
    def forward(self, x):
        B, T, _ = x.shape  # [batch_size, time_length, num_components]
        
        # Apply component-specific embeddings
        embedded = torch.stack([self.embeddings[i](x[:, :, i].unsqueeze(-1)) for i in range(self.num_components)], dim=1)
        # embedded shape: (batch_size, num_components, time_length, embedding_dim)
        
        # Initialize hidden state
        h_0 = torch.zeros(B, 1, self.num_components, self.hidden_dim, device=x.device)
        alignment_matrices = []
        
        for t in range(T):
            # Process one time step
            gru_input = embedded[:, :, t, :].unsqueeze(1).permute(0, 2, 1, 3)  # (batch_size, num_components, 1, embedding_dim)
            gru_input = gru_input.reshape(B*self.num_components, 1, self.embedding_dim) # (batch_size * num_components, 1, embedding_dim)
            h_0 = h_0.permute(1, 0, 2, 3).reshape(1, B*self.num_components, self.hidden_dim) # (1, batch_size * num_components, hidden_dim)
            _, h_0 = self.gru(gru_input, h_0)
            h_0 = h_0.reshape(1, B, self.num_components, self.hidden_dim).permute(1, 0, 2, 3) # (batch_size, 1, num_components, hidden_dim)

            # Reshape h_0 for self-attention
            h_0_reshaped = h_0.squeeze(1)  # (batch_size, num_components, hidden_dim)
            
            # Apply self-attention
            attn_out, attn_output_weights = self.attention(h_0_reshaped, h_0_reshaped, h_0_reshaped)
            
            # Update h_0 with attention output
            h_0 = attn_out.unsqueeze(1)
            
            alignment_matrices.append(attn_output_weights)
        
        # Stack the alignment matrices
        alignment_matrices = torch.stack(alignment_matrices, dim=1)  # (batch_size, seq_len, num_components, num_components)

        attn_input = alignment_matrices.reshape(B, T, -1) # [batch_size; time_length; num_components * num_components]
        ##########################
        DNC_flat = self.gta_attention(attn_input)
        # FC.shape: [batch_size; input_feature_size * input_feature_size]
        ##########################

        # 4. Pass learned graph to the classifier to get predictions
        logits = self.clf(DNC_flat)
        # logits.shape: [batch_size; n_classes]

        return logits, DNC_flat.reshape(B, self.num_components, self.num_components), alignment_matrices

# Example usage
# num_components = 5
# embedding_dim = 32
# hidden_dim = 64
# seq_len = 20
# batch_size = 16

# model = MultivariateTSModel(num_components, embedding_dim, hidden_dim)
# x = torch.randn(batch_size, seq_len, num_components)
# alignment_matrices = model(x)
# print("Alignment matrices shape:", alignment_matrices.shape)