# pylint: disable=too-many-statements, too-many-locals, invalid-name, unbalanced-tuple-unpacking, no-value-for-parameter
"""Script for running experiments: tuning and testing hypertuned models"""
import os
from copy import deepcopy
import math

from omegaconf import OmegaConf, open_dict
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from src.model import model_factory
from src.model_utils import optimizer_factory

from src.datasets.ukb import load_data as load_new
from src.datasets.ukb_old import load_data as load_old
from src.datasets.hcp_roi_752 import load_data as load_hcp
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch
import argparse
import os



class EarlyStopping:
    """Early stops the training if the given score does not improve after a given patience."""

    def __init__(
        self, 
        patience
    ):
        self.counter = 0
        self.patience = patience
        self.early_stop = False
        self.best_score = None

        self.checkpoint = None

    def __call__(self, new_score, model):
        if self.best_score is None:
            self.best_score = new_score
            self.save_checkpoint(model)
        else:
            change = self.best_score - new_score

            if change > 0.0:
                self.counter = 0
                self.best_score = new_score
                self.save_checkpoint(model)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        # based on callback from animus package
        """Saves model if criterion is met"""
        self.checkpoint = deepcopy(model.state_dict())

class InvertedHoyerMeasure(nn.Module):
    """Sparsity loss function based on Hoyer measure: https://jmlr.csail.mit.edu/papers/volume5/hoyer04a/hoyer04a.pdf"""
    def __init__(self, threshold):
        super(InvertedHoyerMeasure, self).__init__()
        self.threshold = threshold
        self.a = nn.LeakyReLU()

    def forward(self, x):
        # Assuming x has shape (batch_size, input_dim, input_dim)

        n = x[0].numel()
        sqrt_n = torch.sqrt(torch.tensor(float(n), device=x.device))

        sum_abs_x = torch.sum(torch.abs(x), dim=(1, 2))
        sqrt_sum_squares = torch.sqrt(torch.sum(torch.square(x), dim=(1, 2)))
        numerator = sqrt_n - sum_abs_x / sqrt_sum_squares
        denominator = sqrt_n - 1
        mod_hoyer = 1 - (numerator / denominator) # = 0 if perfectly sparse, 1 if all are equal
        loss = self.a(mod_hoyer - self.threshold)
        # Calculate the mean loss over the batch
        mean_loss = torch.mean(loss)

        return mean_loss
    
def start(ds, model_name, postfix):
    """Main script for starting experiments"""
    path = "/data/users2/ppopov1/glass_proj/assets/model_weights"
    cfg = OmegaConf.load(f"{path}/dbn_tune_config.yaml")
    with open_dict(cfg):
        cfg.dataset.name = ds
        cfg.model.name = model_name

        cfg.mode.name = "tune"

    os.makedirs(cfg.project_dir+'_'+ds+'_'+model_name, exist_ok=True)

    if ds == "ukb":
        ukb_data, _ = load_new(cfg)
    elif ds == "ukb_old":
        ukb_data, _ = load_old(cfg)
    elif ds == "hcp_roi_752":
        ukb_data, _ = load_hcp(cfg)

    skf = KFold(n_splits=6, shuffle=True, random_state=42)
    CV_folds = list(skf.split(ukb_data))
    train_index, test_index = CV_folds[0]
    tr_data, val_data = ukb_data[train_index], ukb_data[test_index]
    tr_dataloader = DataLoader(torch.tensor(tr_data, dtype=torch.float32), batch_size=cfg.mode.batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(torch.tensor(val_data, dtype=torch.float32), batch_size=cfg.mode.batch_size, num_workers=0)

    with open(f"{cfg.project_dir}/general_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(cfg, f)

    model_cfg = OmegaConf.create({
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
            "pred_weight": 1.0,
        },
        "lr": 1e-3,
        "load_pretrained": False,
        "input_size": 53,
        "output_size": 2,
    })

    if torch.cuda.is_available():
        # CUDA
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    
    model = model_factory(cfg, model_cfg).to(device)

    optimizer = optimizer_factory(cfg, model_cfg, model)
    LR_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)
    early_stopper = EarlyStopping(cfg.mode.max_epochs)
    
    for epoch in range(cfg.mode.max_epochs):
        
        train_loss = 0.0
        train_loss_components = {} 
        with torch.set_grad_enabled(True):
            model.train()
            n_batches = math.ceil(len(tr_dataloader.dataset) / tr_dataloader.batch_size)
            for targets in tr_dataloader:
                targets = targets.to(device)

                output = model(targets, pretraining=True)
                loss, loss_logs = model.compute_loss(output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                for key, value in loss_logs.items():
                    train_loss_components[key] = train_loss_components.get(key, 0.0) + value
        
        
        val_loss = 0.0
        val_loss_components = {} 
        with torch.set_grad_enabled(False):
            n_batches_val = math.ceil(len(val_dataloader.dataset) / val_dataloader.batch_size)
            model.eval()
            for targets in val_dataloader:
                targets = targets.to(device)

                output = model(targets, pretraining=True)
                loss, loss_logs = model.compute_loss(output)

                val_loss += loss.item()
                for key, value in loss_logs.items():
                    val_loss_components[key] = val_loss_components.get(key, 0.0) + value

        early_stopper(val_loss, model)
        LR_scheduler.step(val_loss)

        train_loss_components = {key: value / n_batches for key, value in train_loss_components.items()}
        val_loss_components = {key: value / n_batches_val for key, value in val_loss_components.items()}
        log = {
            "epoch": epoch,
            "tr_loss": train_loss/n_batches,
            "val_loss": val_loss/n_batches_val,
            **{f"tr_{key}": value for key, value in train_loss_components.items()},
            **{f"val_{key}": value for key, value in val_loss_components.items()},

        }

        if not os.path.isfile(f"{path}/log_{ds}_{model_name}_{postfix}.csv"):
            pd.DataFrame([log]).to_csv(f"{path}/log_{ds}_{model_name}_{postfix}.csv", index=False)
        else:
            # Append to the existing file without writing the header
            pd.DataFrame([log]).to_csv(f"{path}/log_{ds}_{model_name}_{postfix}.csv", mode='a', header=False, index=False)

        print(log)

    torch.save(early_stopper.checkpoint, f"{path}/{model_name}_{ds}_{postfix}.pt")

    target = next(iter(val_dataloader))
    target = target.to(device)
    with torch.set_grad_enabled(False):
        model.eval()
        output = model(target, pretraining=True)
        predicted = output["predicted"]
        origs = output["originals"]

    
    mse_diffs = torch.mean((predicted - origs) ** 2, dim=[1, 2])

    smallest_indices = torch.topk(mse_diffs, 6, largest=False).indices  # Indices for smallest diffs
    largest_indices = torch.topk(mse_diffs, 6, largest=True).indices    # Indices for largest diffs

    # Function to plot a 2D tensor as a colormap
    def plot_heatmap(ax, data, title, cmap='seismic', vmin=None, vmax=None):
        cax = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(cax, ax=ax)

    # Step 3: Plot the 4x3 grid (3 sets of data per 2 examples)
    fig, axes = plt.subplots(5, 12, figsize=(44, 16))

    # Helper function to plot rows for both smallest and largest indices
    def plot_rows(tensor1, tensor2, indices, row_offset, axes):
        for i, idx in enumerate(indices):
            # Row 1: Reconstructed (tensor1)
            vmax1 = torch.abs(tensor1).max().item()

            # Row 2: Original (tensor2)
            vmax2 = torch.abs(tensor2).max().item()
            vmax = max([vmax1, vmax2])
            plot_heatmap(axes[0, i + row_offset], tensor1[idx].cpu().detach().numpy().T, title=f'Pred {idx.item()}', vmin=-vmax, vmax=vmax)
            plot_heatmap(axes[1, i + row_offset], tensor2[idx].cpu().detach().numpy().T, title=f'Orig {idx.item()}', vmin=-vmax, vmax=vmax)

            # Row 3: Difference (tensor1 - tensor2)
            diff1 = tensor1[idx] - tensor2[idx]
            vmax1 = torch.abs(diff1).max().item()  # Set vmin/vmax based on the difference

            # Row 4: 
            diff2 = tensor1[idx, 1:] - tensor2[idx, :-1]
            vmax2 = torch.abs(diff2).max().item()  # Set vmin/vmax based on the difference

            # Row 5: 
            diff3 = tensor2[idx, 1:] - tensor2[idx, :-1]
            vmax3 = torch.abs(diff3).max().item()  # Set vmin/vmax based on the difference

            vmax = max(vmax1, vmax2, vmax3)

            plot_heatmap(axes[2, i + row_offset], diff1.cpu().detach().numpy().T, title=f'Pred - Orig, MSE={torch.mean(diff1**2).item():04f}', vmin=-vmax, vmax=vmax)
            plot_heatmap(axes[3, i + row_offset], diff2.cpu().detach().numpy().T, title=f'Pred_t - Orig_(t-1), MSE={torch.mean(diff2**2).item():04f}', vmin=-vmax, vmax=vmax)
            plot_heatmap(axes[4, i + row_offset], diff3.cpu().detach().numpy().T, title=f'Orig_t - Orig_(t-1), MSE={torch.mean(diff3**2).item():04f}', vmin=-vmax, vmax=vmax)

    # Step 4: Plot the smallest and largest difference pairs
    plot_rows(predicted, origs, smallest_indices, row_offset=0, axes=axes)  # Smallest differences on the left 2 columns
    plot_rows(predicted, origs, torch.flip(largest_indices, [0]), row_offset=6, axes=axes)   # Largest differences on the right 2 columns

    plt.tight_layout()
    plt.savefig(f"{path}/recon_{ds}_{model_name}_{postfix}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single string input.')

    # Add a string argument
    parser.add_argument('ds', type=str, help='The input string to process.')

    parser.add_argument('model', type=str, help='The input string to process.')
    parser.add_argument('postfix', type=str, help='The input string to process.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the provided argument
    start(args.ds, args.model, args.postfix)
