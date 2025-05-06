# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DECIFRA model """

import os

import torch
from torch import nn
from torch.nn import functional as F

from omegaconf import OmegaConf, DictConfig

def default_HPs(input_size, n_classes):
    model_cfg = {
        "rnn": {
            "input_embedding_size": 16,
            "hidden_size": 16,
        },
        "btp": {
            "hidden_dim": 16,
        },
        "loss": {
            "threshold": 0.01,
            "sp_weight": 1.0, # 1.0 for pretraining, 0.02 for experiments with classification
            "forecast_weight": 1.0, # 1.0 for pretraining, 0.02 for experiments with classification
        },
        "lr": 1e-4,
        "load_pretrained": False,
        "pretrained_path": None,
        "input_size": input_size, # input feature size at each time point
        "output_size": n_classes,
    }
    return OmegaConf.create(model_cfg)

def get_model(model_cfg: DictConfig):
    model = DECIFRA(model_cfg)
    if model_cfg.load_pretrained:
        assert model_cfg.pretrained_path is not None, "Error while loading model, load_pretrained is True, yet pretrained path is not set in the config."

        path = model_cfg.pretrained_path
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        dont_load = ["clf"]
        pruned_checkpoint = {k: v for k, v in checkpoint.items() if not any(key in k for key in dont_load)}
        model.load_state_dict(pruned_checkpoint, strict=False)

    return model


class DECIFRALoss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg):
        self.sparsity_loss = InvertedHoyerMeasure(threshold=model_cfg.loss.threshold)

        self.sp_weight = model_cfg.loss.sp_weight
        self.forecast_weight = model_cfg.loss.forecast_weight


    def __call__(self, logits, target, matrices, predicted, originals):
        B, T, C, _ = matrices.shape
        matrices = matrices.reshape(B*T, C, C)
        sparse_loss = self.sparsity_loss(matrices)

        forecast_loss = F.mse_loss(predicted, originals)

        if logits is not None and target is not None: # training case
            ce_loss = F.cross_entropy(logits, target)

            loss = ce_loss + self.sp_weight * sparse_loss + self.forecast_weight * forecast_loss

            loss_components = {
                "ce_loss": ce_loss.item(),
                "sp_loss": sparse_loss.item(),
                "forecast_loss": forecast_loss.item(),
            }
        
        else: # pretraining case
            loss = self.sp_weight * sparse_loss + self.pred_weight * forecast_loss

            loss_components = {
                "sp_loss": sparse_loss.item(),
                "forecast_loss": forecast_loss.item(),
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

class DECIFRA(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(DECIFRA, self).__init__()

        self.input_size = input_size = model_cfg.input_size # n_components (#ROIs/ICs/channels)
        self.embedding_dim = embedding_dim = model_cfg.rnn.input_embedding_size # embedding size for GRU input
        self.hidden_dim = hidden_dim = model_cfg.rnn.hidden_size # GRU hidden dim
        output_size = model_cfg.output_size # n_classes to predict

        # GRU and its input embedding
        self.embeddings = nn.Linear(1, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # bilinear transition predictor (BTP) used to compute the transfer matrices
        self.BTP = BTP(
            input_dim=hidden_dim, 
            hidden_dim=model_cfg.btp.hidden_dim,
            n_components=self.input_size
        )

        # Classifier decoder
        self.clf = nn.Sequential(
            nn.Linear(input_size**2, input_size**2 // 2),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
            nn.Linear(input_size**2 // 2, input_size**2 // 4),
            nn.ReLU(),
            nn.Linear(input_size**2 // 4, output_size),
        )
        # Forecasting decoder
        self.predictor = nn.Linear(hidden_dim, 1)

        self.criterion = DECIFRALoss(model_cfg)

    def compute_loss(self, additional_outputs, logits, target):
        loss, log = self.criterion(
            logits=logits, 
            target=target, 
            matrices=additional_outputs["matrices"], 
            predicted=additional_outputs["predicted"],
            originals=additional_outputs["originals"]
        )

        return loss, log

    
    def save_data(self, cfg, ds_name, data, target, additional_outputs):
        save_path = f"{cfg.run_dir}/data"
        os.makedirs(save_path, exist_ok=True)
        torch.save(data, f"{save_path}/{ds_name}_input.pt")
        torch.save(target, f"{save_path}/{ds_name}_labels.pt")
        torch.save(additional_outputs["matrices"], f"{save_path}/{ds_name}_matrices.pt")
        torch.save(additional_outputs["time_logits"], f"{save_path}/{ds_name}_time_logits.pt")
        if "holdout" in ds_name:
            plot_combined_matrices(additional_outputs["matrices"], f"{save_path}/{ds_name}_time_matrices.png", n_samples=1)
            plot_mean_matrices(additional_outputs["matrices"], f"{save_path}/{ds_name}_mean_matrices.png", n_samples=-1)

    def forward(self, x, pretraining=False):
        B, T, C = x.shape  # [batch_size, time_length, input_size==self.input_size]
        orig_x = x

        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, T, 1)
        embedded = self.embeddings(x) # shape: (B, C, T, self.embedding_dim)

        
        # Initialize hidden state and run the recurren loop
        h = torch.zeros(B, C, self.hidden_dim, device=x.device)
        # hidden state shape: [B, C, self.hidden_dim]

        transfer_matrices = []
        hidden_states = []
        for t in range(T):
            # Process one time step
            gru_input = embedded[:, :, t, :].unsqueeze(2)  # (B, C, 1, embedding_dim)
            gru_input = gru_input.reshape(B*C, 1, self.embedding_dim) # (B*C, 1, embedding_dim)
            # input hidden state must have shape (D * num_layers, N, hidden_size), D*num_layers = 1 in our case, N is effective GRU batch size
            h = h.reshape(1, B*C, self.hidden_dim) # (1, B*C, hidden_dim)

            # update the hidden states with the new input by running GRU
            _, h = self.gru(gru_input, h) # output h shape is the same: (1, GRU_batch, hidden_size)
            h = h.reshape(B, self.input_size, self.hidden_dim) # (B, C, hidden_dim)

            # Find transition matrix and mix hidden states
            h, mixing_matrix = self.BTP(h)
            hidden_states.append(h)
            transfer_matrices.append(mixing_matrix)

            if torch.any(torch.isnan(h)): # for debugging, h will have invalid values if model diverges
                raise Exception(f"h has nans at time point {t}")

        # Stack the alignment matrices, predict the next input 
        transfer_matrices = torch.stack(transfer_matrices, dim=1)  # (batch_size, seq_len, input_size, input_size)
        hidden_states = torch.stack(hidden_states, dim=1)[:, :-1, :, :] # brain latent states starting with time 0, [batch_size; time_length-1; input_size, hidden_dim]
        predicted = self.predictor(hidden_states).squeeze() # predictions of x starting with time 1, [batch_size; time_length-1; input_size]
        
        if pretraining:
            # pretrain on the forecasting task
            return None, {
                "matrices": transfer_matrices,
                "predicted": predicted,
                "originals": orig_x[:, 1:, :]
            }
        
        clf_input = transfer_matrices.reshape(B, T, -1) # [batch_size; time_length; input_size * input_size]
        time_logits = self.clf(clf_input) # [batch_size; time_length, n_classes]
        logits = torch.mean(time_logits, dim=1) # mean over time, [batch_size; n_classes]

        additional_outputs = {
            "matrices": transfer_matrices,
            "time_logits": time_logits,
            "predicted": predicted,
            "originals": orig_x[:, 1:, :]
        }

        return logits, additional_outputs



class BTP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_components):
        super(BTP, self).__init__()
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
    
### a few utility functions for transfer matrix evaludation

import matplotlib.pyplot as plt
import numpy as np
def plot_combined_matrices(matrices, save_path, n_samples=5, n_time=5):
    if n_samples == -1:
         n_samples = matrices.shape[0]
         
    # Normalize the range for the seismic colormap to center around 0
    abs_max = matrices[:n_samples, 50:(n_time+50)].abs().max().item()
    vmin, vmax = -abs_max, abs_max  # Centering colormap around 0

    # Determine the size of individual matrices
    matrix_size = matrices[0, 0].shape[0]

    # Create a large matrix to hold all the smaller matrices with padding
    combined_matrix = np.full(
        ((matrix_size + 1) * n_samples - 1, (matrix_size + 1) * n_time - 1),
        0.1
    )

    for i in range(n_samples):
        for j in range(n_time):
            matrix = matrices[i, 50 + j].cpu().detach().numpy()  # Convert to NumPy
            start_row = i * (matrix_size + 1)
            start_col = j * (matrix_size + 1)
            combined_matrix[start_row:start_row + matrix_size, start_col:start_col + matrix_size] = matrix

    # Plot the combined matrix
    # dpi=1
    dpi=4
    figsize = combined_matrix.shape[1] * dpi, combined_matrix.shape[0] * dpi  # Match the figure size to the array dimensions (pixels)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(combined_matrix, cmap="seismic", vmin=vmin, vmax=vmax, interpolation='none')
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def plot_mean_matrices(matrices, save_path, n_samples=5):
    ### also plot mean-over-time matrices ###
    if n_samples == -1:
         n_samples = matrices.shape[0]
    # Calculate the mean over time for each sample
    mean_matrices = matrices[:n_samples].mean(dim=1)

    # Normalize the range for the seismic colormap to center around 0
    abs_max = mean_matrices.abs().max().item()
    vmin, vmax = -abs_max, abs_max  # Centering colormap around 0

    matrix_size = matrices[0, 0].shape[0]
    combined_matrix = np.full(
        ((matrix_size + 1) * n_samples - 1, matrix_size),
        0.1
    )
    for i in range(n_samples):
            matrix = mean_matrices[i].cpu().detach().numpy()  # Convert to NumPy
            start_row = i * (matrix_size + 1)
            combined_matrix[start_row:start_row + matrix_size, :] = matrix
            
    # Plot the combined matrix
    dpi=4
    figsize = combined_matrix.shape[1] * dpi, combined_matrix.shape[0] * dpi  # Match the figure size to the array dimensions (pixels)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(combined_matrix, cmap="seismic", vmin=vmin, vmax=vmax, interpolation='none')
    plt.savefig(save_path, dpi=dpi)
    plt.close()