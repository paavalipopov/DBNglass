# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DICE model from https://github.com/UsmanMahmood27/DICE """
from random import uniform, randint

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from omegaconf import OmegaConf, DictConfig

import ipdb


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return MultivariateTSModel(model_cfg)


def get_criterion(cfg: DictConfig, model_cfg: DictConfig):
    return RegCEloss(model_cfg)

class InvertedHoyerMeasure(nn.Module):
    """Sparsity loss function based on Hoyer measure: https://jmlr.csail.mit.edu/papers/volume5/hoyer04a/hoyer04a.pdf"""
    def __init__(self, threshold):
        super(InvertedHoyerMeasure, self).__init__()
        self.threshold = threshold
        self.a = nn.LeakyReLU()

    def forward(self, x):
        # Assuming x has shape (batch_size, input_dim, input_dim)
        batch_size = x.size(0)
        losses = []

        for i in range(batch_size):
            # Extract the ith sample
            xi = x[i]
            
            # Number of elements in the sample
            n = xi.numel()
            sqrt_n = torch.sqrt(torch.tensor(float(n), device=xi.device))

            # Sum of absolute values
            sum_abs_x = torch.sum(torch.abs(xi))

            # Square root of the sum of squares
            sqrt_sum_squares = torch.sqrt(torch.sum(xi ** 2))

            # Calculate the Hoyer measure
            numerator = sqrt_n - (sum_abs_x / sqrt_sum_squares)
            denominator = sqrt_n - 1
            mod_hoyer = 1 - (numerator / denominator) # = 0 if perfectly sparse, 1 if all are equal
            
            loss = self.a(mod_hoyer - self.threshold)

            losses.append(loss)

        # Calculate the mean loss over the batch
        mean_loss = torch.mean(torch.stack(losses))

        return mean_loss


class RegCEloss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg, lambda1 = 0.01):
        self.ce_loss = nn.CrossEntropyLoss()
        self.sparsity_loss = InvertedHoyerMeasure(threshold=model_cfg.loss.threshold)

        self.minimize_global = model_cfg.loss.minimize_global

        # self.lambda1 = model_cfg.loss.lambda1 # Sparsity loss weight

    def __call__(self, logits, target, model, device, DNC, DNCs):
        ce_loss = self.ce_loss(logits, target)

        # Sparsity loss on DNC
        if self.minimize_global:
            sparse_loss =  self.sparsity_loss(DNC)
        else:
            B, T, C, _ = DNCs.shape
            DNCs = DNCs.reshape(B*T, C, C)
            sparse_loss = self.sparsity_loss(DNCs)

        # loss = ce_loss + self.lambda1 * sparse_loss
        loss = ce_loss + sparse_loss
        return loss
    


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "rnn": {
            "single_embed": True,
            "num_layers": 1,
            "input_embedding_size": 32,
            "hidden_embedding_size": 64,
        },
        "attention": {
            "hidden_dim": 32,
            "track_grads": True,
            "use_tan": "none",
            "use_gate": True,
        },
        "loss": {
            "minimize_global": False,
            "threshold": 0.1,
        },
        "lr": 4e-5,
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
            "hidden_embedding_size": optuna_trial.suggest_int("rnn.hidden_embedding_size", 16, 128),
        },
        "attention": {
            "hidden_dim": optuna_trial.suggest_int("attention.hidden_dim", 4, 64),
            "track_grads": True,
            "use_tan": "none",
            "use_gate": True,
        },
        "loss": {
            "minimize_global": False,
            "threshold": 10 ** optuna_trial.suggest_float("loss.threshold", -2, -0.2),
        },
        "lr": 10 ** optuna_trial.suggest_float("lr", -5, -3),
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)

class MultivariateTSModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(MultivariateTSModel, self).__init__()

        self.num_components = num_components = model_cfg.input_size
        self.num_layers = num_layers = model_cfg.rnn.num_layers
        self.embedding_dim = embedding_dim = model_cfg.rnn.input_embedding_size
        self.hidden_dim = hidden_dim = model_cfg.rnn.hidden_embedding_size
        output_size = model_cfg.output_size

        self.single_embed = model_cfg.rnn.single_embed
        
        # Component-specific embeddings
        if model_cfg.rnn.single_embed:
            self.embeddings = nn.Linear(1, embedding_dim)
        else:
            self.embeddings = nn.ModuleList([
                nn.Linear(1, embedding_dim) for _ in range(num_components)
            ])


        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # Self-attention layer
        self.attention = SelfAttention(
            input_dim=hidden_dim, 
            hidden_dim=model_cfg.attention.hidden_dim, 
            track_grads=model_cfg.attention.track_grads,
            use_tan=model_cfg.attention.use_tan,
            use_gate=model_cfg.attention.use_gate,
            n_components=self.num_components
        )

        # Global Temporal Attention 
        self.upscale = 0.05
        self.upscale2 = 0.5

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
        self.clf = nn.Sequential(
            nn.Linear(num_components**2, num_components**2 // 2),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
            nn.Linear(num_components**2 // 2, num_components**2 // 4),
            nn.ReLU(),
            nn.Linear(num_components**2 // 4, output_size),
        )

    def gta_attention(self, x, node_axis=1):
        # x.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        x_readout = x.mean(node_axis, keepdim=True)
        x_readout = x * x_readout

        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1, x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
        x_graphattention = F.softmax(x_graphattention, dim=1)
        return (x * (x_graphattention.unsqueeze(-1))).sum(node_axis)
    
        # x_graphattention = self.HW(x_graphattention.reshape(a, b))
        # return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)
    
    def dump_data(self, data, path, basename):
        for i, dat in enumerate(data):
            torch.save(dat, f"{path}/{basename}_{i}.pt")

    def forward(self, x):
        savepath = "/data/users2/ppopov1/glass_proj/scripts/debug2/"
        B, T, _ = x.shape  # [batch_size, time_length, num_components]
        
        # Apply component-specific embeddings
        # embedded = torch.stack([self.embeddings[i](x[:, :, i].unsqueeze(-1)) for i in range(self.num_components)], dim=1)
        if self.single_embed:
            x = x.permute(0, 2, 1)
            x = x.reshape(B * self.num_components, T, 1)
            embedded = self.embeddings(x).reshape(B, self.num_components, T, self.embedding_dim)
        else:
            embedded = torch.stack([self.embeddings[i](x[:, :, i].unsqueeze(-1)) for i in range(self.num_components)], dim=1)

        if torch.any(torch.isnan(x)):
            print("X has nons")
        if torch.any(torch.isnan(embedded)):
            print("embedded has nons")
        
        # Initialize hidden state
        h_0 = torch.zeros(B, 1, self.num_components, self.hidden_dim, device=x.device)
        torch.save(h_0, savepath+"h_init.pt")

        alignment_matrices = []
        h_0_gru = []
        h_0_attn = []
        
        for t in range(T):
            # Process one time step
            gru_input = embedded[:, :, t, :].unsqueeze(1).permute(0, 2, 1, 3)  # (batch_size, num_components, 1, embedding_dim)
            gru_input = gru_input.reshape(B*self.num_components, 1, self.embedding_dim) # (batch_size * num_components, 1, embedding_dim)
            h_0 = h_0.permute(1, 0, 2, 3).reshape(1, B*self.num_components, self.hidden_dim) # (1, batch_size * num_components, hidden_dim)
            _, h_0 = self.gru(gru_input, h_0)
            h_0 = h_0.reshape(1, B, self.num_components, self.hidden_dim).permute(1, 0, 2, 3) # (batch_size, 1, num_components, hidden_dim)
            # h_0_gru.append(h_0)

            # Reshape h_0 for self-attention
            h_0_reshaped = h_0.squeeze(1)  # (batch_size, num_components, hidden_dim)

            # Apply self-attention
            attn_out, attn_output_weights = self.attention(h_0_reshaped)

            # Update h_0 with attention output
            h_0 = attn_out.unsqueeze(1)
            # h_0_attn.append(h_0)

            alignment_matrices.append(attn_output_weights)

            if torch.any(torch.isnan(h_0)):
                # self.dump_data(alignment_matrices, savepath, "align_matrix")
                # self.dump_data(h_0_gru, savepath, "h_gru")
                # self.dump_data(h_0_attn, savepath, "h_attn")
                
                raise Exception(f"h_0 has nans at time point {t}")
            
        
        # Stack the alignment matrices
        alignment_matrices = torch.stack(alignment_matrices, dim=1)  # (batch_size, seq_len, num_components, num_components)
        
        attn_input = alignment_matrices.reshape(B, T, -1) # [batch_size; time_length; num_components * num_components]

        DNC_flat = self.gta_attention(attn_input)
        
        # 4. Pass learned graph to the classifier to get predictions
        logits = self.clf(DNC_flat)
        # logits.shape: [batch_size; n_classes]

        return logits, DNC_flat.reshape(B, self.num_components, self.num_components), alignment_matrices



class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, track_grads, use_tan, use_gate, n_components):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.track_grads = track_grads
        self.use_tan = use_tan
        self.use_gate = use_gate

        if use_gate:
            self.gate = Gate(n_components)

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)


    def forward(self, x): # x.shape (batch_size, seq_length, input_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2))

        if self.use_tan == "before":
            scores = F.tanh(scores)

        # eigenvals = torch.linalg.eigvals(scores)
        eigenvals = torch.linalg.svdvals(scores)
        eigenvals = torch.max(torch.abs(eigenvals), dim=1)[0].view(x.shape[0], 1, 1)
        if not self.track_grads:
            eigenvals = eigenvals.detach()
        scores = scores / eigenvals

        if self.use_tan == "after":
            scores = F.tanh(scores)

        attention = scores
        if self.use_gate:
            gate = self.gate(attention)
            attention = attention * gate

        weighted = torch.bmm(attention, values)

        return weighted, attention

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