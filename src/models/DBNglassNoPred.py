# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" glassDBN model """

import os

import torch
from torch import nn
from torch.nn import functional as F

from omegaconf import OmegaConf, DictConfig
from src.settings import WEIGHTS_ROOT

def get_model(cfg: DictConfig, model_cfg: DictConfig):
    model = glassDBN(model_cfg)
    if model_cfg.load_pretrained == True:
        path = model_cfg.pretrained_path
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        dont_load = ["clf"]
        pruned_checkpoint = {k: v for k, v in checkpoint.items() if not any(key in k for key in dont_load)}
        model.load_state_dict(pruned_checkpoint, strict=False)

    return model


class RegCEloss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg):
        self.sparsity_loss = InvertedHoyerMeasure(threshold=model_cfg.loss.threshold)

        self.sp_weight = model_cfg.loss.sp_weight


    def __call__(self, logits, target, DNCs):
        ce_loss = F.cross_entropy(logits, target)

        B, T, C, _ = DNCs.shape
        DNCs = DNCs.reshape(B*T, C, C)
        sparse_loss = self.sparsity_loss(DNCs)

        loss = ce_loss + self.sp_weight * sparse_loss

        loss_components = {
            "ce_loss": ce_loss.item(),
            "sp_loss": sparse_loss.item(),
        }
        return loss, loss_components

class InvertedHoyerMeasure:
    """Sparsity loss function based on Hoyer measure: https://jmlr.csail.mit.edu/papers/volume5/hoyer04a/hoyer04a.pdf"""
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x):
        # Assuming x has shape (batch_size, input_dim, input_dim)

        n = x[0].numel()
        sqrt_n = torch.sqrt(torch.tensor(float(n), device=x.device))
        sum_abs_x = torch.sum(torch.abs(x), dim=(1, 2))
        sqrt_sum_squares = torch.sqrt(torch.sum(torch.square(x), dim=(1, 2)))

        numerator = sqrt_n - sum_abs_x / sqrt_sum_squares
        denominator = sqrt_n - 1
        mod_hoyer = 1 - (numerator / denominator) # = 0 if perfectly sparse, 1 if all are equal

        loss = F.leaky_relu(mod_hoyer - self.threshold)
        # Calculate the mean loss over the batch
        mean_loss = torch.mean(loss)

        return mean_loss
    

def default_HPs(cfg: DictConfig):
    pretrained = cfg.pretrained

    model_cfg = {
        "rnn": {
            "single_embed": True,
            "num_layers": 1,
            "input_embedding_size": 16,
            "hidden_size": 16,
        },
        "attention": {
            "hidden_dim": 16,
        },
        "loss": {
            "threshold": 0.01,
            "sp_weight": 1.0,
        },
        "lr": 1e-4,
        "load_pretrained": pretrained,
        # "load_pretrained": True,
        "pretrained_path": str(WEIGHTS_ROOT.joinpath(f"DBNglassFIX_ukb.pt")),
        # "pretrained_path": str(WEIGHTS_ROOT.joinpath(f"DBNglassFIX_ukb_{cfg.idx}.pt")),
        # "pretrained_path": path,
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig, optuna_trial=None):
    model_cfg = {
        "rnn": {
            "single_embed": True,
            "num_layers": 1,
            "input_embedding_size": optuna_trial.suggest_int("rnn.input_embedding_size", 4, 64),
            "hidden_size": optuna_trial.suggest_int("rnn.hidden_embedding_size", 4, 128),
        },
        "attention": {
            "hidden_dim": optuna_trial.suggest_int("attention.hidden_dim", 4, 64),
        },
        "loss": {
            "minimize_global": False,
            "threshold": 10 ** optuna_trial.suggest_float("loss.threshold", -2, -0.2),
            "lambdaa": 10 ** optuna_trial.suggest_float("loss.threshold", -1, 1),
        },
        "lr": 10 ** optuna_trial.suggest_float("lr", -5, -3),
        "load_pretrained": False,
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)

class glassDBN(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(glassDBN, self).__init__()

        self.input_size = input_size = model_cfg.input_size # n_components (#ROIs/ICs)
        self.num_layers = num_layers = model_cfg.rnn.num_layers # GRU n_layers; won't work with values other than 1
        self.embedding_dim = embedding_dim = model_cfg.rnn.input_embedding_size # embedding size for GRU input
        self.hidden_dim = hidden_dim = model_cfg.rnn.hidden_size # GRU hidden dim
        output_size = model_cfg.output_size # n_classes to predict
        self.single_embed = model_cfg.rnn.single_embed # whether all time series should be embedded with the same vector or not
        
        # Component-specific embeddings
        if model_cfg.rnn.single_embed:
            self.embeddings = nn.Linear(1, embedding_dim)
        else:
            self.embeddings = nn.ModuleList([
                nn.Linear(1, embedding_dim) for _ in range(input_size)
            ])

        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # Attention layer used to compute the transfer matrices
        self.attention = SelfAttention(
            input_dim=hidden_dim, 
            hidden_dim=model_cfg.attention.hidden_dim,
            n_components=self.input_size
        )

        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(input_size**2, input_size**2 // 2),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
            nn.Linear(input_size**2 // 2, input_size**2 // 4),
            nn.ReLU(),
            nn.Linear(input_size**2 // 4, output_size),
        )
        # Input predictor
        self.predictor = nn.Linear(hidden_dim, 1)

        self.criterion = RegCEloss(model_cfg)

    def compute_loss(self, logits, target, additional_outputs):
        loss, log = self.criterion(
            logits=logits, 
            target=target, 
            DNCs=additional_outputs["DNCs"],
        )

        return loss, log

    
    def save_data(self, cfg, ds_name, data, target, additional_outputs):
        save_path = f"{cfg.run_dir}/data"
        os.makedirs(save_path, exist_ok=True)
        torch.save(data, f"{save_path}/{ds_name}_input.pt")
        torch.save(target, f"{save_path}/{ds_name}_labels.pt")
        torch.save(additional_outputs["DNCs"], f"{save_path}/{ds_name}_DNCs.pt")
        torch.save(additional_outputs["time_logits"], f"{save_path}/{ds_name}_time_logits.pt")

    def forward(self, x, pretraining=False):
        B, T, _ = x.shape  # [batch_size, time_length, input_size]
        orig_x = x

        # Apply component-specific embeddings
        if self.single_embed:
            x = x.permute(0, 2, 1)
            x = x.reshape(B * self.input_size, T, 1)
            embedded = self.embeddings(x).reshape(B, self.input_size, T, self.embedding_dim)
        else:
            embedded = torch.stack([self.embeddings[i](x[:, :, i].unsqueeze(-1)) for i in range(self.input_size)], dim=1)
        # embedded shape: [batch_size, input_size, time_length, embedding_dim]
        
        # Initialize hidden state and run the recurren loop
        h = torch.zeros(B, 1, self.input_size, self.hidden_dim, device=x.device)
        # hidden state shape: [batch_size, 1, input_size, hidden_dim]

        mixing_matrices = []
        hidden_states = []
        for t in range(T):
            # Process one time step
            gru_input = embedded[:, :, t, :].unsqueeze(2)  # (batch_size, input_size, 1, embedding_dim)
            gru_input = gru_input.reshape(B*self.input_size, 1, self.embedding_dim) # (batch_size * input_size, 1, embedding_dim)
            h = h.permute(1, 0, 2, 3).reshape(1, B*self.input_size, self.hidden_dim) # (1, batch_size * input_size, hidden_dim)
            _, h = self.gru(gru_input, h)
            h = h.reshape(1, B, self.input_size, self.hidden_dim).permute(1, 0, 2, 3) # (batch_size, 1, input_size, hidden_dim)

            # Reshape h for self-attention
            h = h.squeeze(1)  # (batch_size, input_size, hidden_dim)
            # Apply self-attention
            h, mixing_matrix = self.attention(h)
            hidden_states.append(h)
            mixing_matrices.append(mixing_matrix)
            h = h.unsqueeze(1) # (batch_size, 1, input_size, hidden_dim)

            if torch.any(torch.isnan(h)):
                raise Exception(f"h has nans at time point {t}")
            
        
        # Stack the alignment matrices, predict the next input 
        mixing_matrices = torch.stack(mixing_matrices, dim=1)  # (batch_size, seq_len, input_size, input_size)
        hidden_states = torch.stack(hidden_states, dim=1)[:, :-1, :, :] # brain latent states starting with time 0, [batch_size; time_length-1; input_size, hidden_dim]
        predicted = self.predictor(hidden_states).squeeze() # predictions of x starting with time 1, [batch_size; time_length-1; input_size]
        
        if pretraining:
            # pretrain on the input prediction task
            return {
                "DNCs": mixing_matrices,
                "predicted": predicted,
                "originals": orig_x[:, 1:, :],
            }
        
        clf_input = mixing_matrices.reshape(B, T, -1) # [batch_size; time_length; input_size * input_size]
        time_logits = self.clf(clf_input) # [batch_size; time_length, n_classes]
        logits = torch.mean(time_logits, dim=1) # mean over time, [batch_size; n_classes]
        
        additional_outputs = {
            "DNCs": mixing_matrices,
            "time_logits": time_logits,
            "predicted": predicted,
            "originals": orig_x[:, 1:, :]
        }

        return logits, additional_outputs



class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim

        self.gate = Gate(n_components)

        self.query = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.key = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


    def forward(self, x): # x.shape (batch_size, n_components, GRU hidden size)
        queries = self.query(x)
        keys = self.key(x)

        transfer = torch.bmm(queries, keys.transpose(1, 2))
        norms = torch.linalg.matrix_norm(transfer, keepdim=True)
        transfer = transfer / norms

        gate = self.gate(transfer)
        transfer = transfer * gate

        next_states = torch.bmm(transfer, x)

        return next_states, transfer

class Gate(nn.Module):
    def __init__(self, input_dim):
        super(Gate, self).__init__()
        self.bias = nn.Parameter(torch.randn(input_dim, input_dim))
    
    def forward(self, x):
        # Compute h_ij = abs(x_ij) + b_ij
        h = torch.abs(x) + self.bias
        
        # Compute a_ij = sigmoid(h_ij)
        a = torch.sigmoid(h)
        
        return a