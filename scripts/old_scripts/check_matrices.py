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
from src.data import data_factory, data_postfactory
from src.dataloader import dataloader_factory, cross_validation_split

from torch import nn, optim
import torch
import argparse
import os

    
def start():
    """Main script for starting experiments"""
    # path = "/data/users2/ppopov1/glass_proj/assets/logs/_fix_rerun-exp-DBNglassFIX_defHP-fbirn/k_00/trial_0009/config.yaml"
    # model_path = "/data/users2/ppopov1/glass_proj/assets/logs/_fix_rerun-exp-DBNglassFIX_defHP-fbirn/k_00/trial_0009/model_config.yaml"
    path = "/data/users2/ppopov1/glass_proj/assets/logs/_repretrained-exp-DBNglassFIX_defHP-fbirn/k_00/trial_0009/config.yaml"
    model_path = "/data/users2/ppopov1/glass_proj/assets/logs/_repretrained-exp-DBNglassFIX_defHP-fbirn/k_00/trial_0009/model_config.yaml"
    # path = "/data/users2/ppopov1/glass_proj/assets/logs/_sparse-exp-DBNglassFIX_defHP-fbirn/k_00/trial_0009/config.yaml"
    # model_path = "/data/users2/ppopov1/glass_proj/assets/logs/_sparse-exp-DBNglassFIX_defHP-fbirn/k_00/trial_0009/model_config.yaml"
    cfg = OmegaConf.load(path)
    model_cfg = OmegaConf.load(model_path)
    
    if torch.cuda.is_available():
        # CUDA
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    model = model_factory(cfg, model_cfg).to(device)
    original_data = data_factory(cfg)
    dataloaders = dataloader_factory(cfg, original_data, k=0, trial=9)
    dataloader = dataloaders["test"]
    
    matrices = []
    with torch.set_grad_enabled(False):
        for x, y in dataloader:
            x = x.to(device)
            mixing_matrices, _, _ = model(x, pretraining=True)
            matrices.append(mixing_matrices.cpu().detach())
    matrix = matrices[0]
    
    torch.save(matrix,  "/data/users2/ppopov1/glass_proj/assets/utility_logs/DNCs_repretrained.pt")
    # torch.save(matrix,  "/data/users2/ppopov1/glass_proj/assets/utility_logs/DNCs_sparse.pt")

if __name__ == "__main__":
    start()
