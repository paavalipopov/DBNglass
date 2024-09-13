# pylint: disable=too-many-function-args, invalid-name
""" UKB ICA (with age bins (X) sex) labels dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.datasets.ukb import load_data as load_sex_data


def load_data(
    cfg: DictConfig,
    dataset_path: str = "/data/users2/ppopov1/datasets/ukb/UKB_age_data.npz",
    indices_path: str = "/data/users2/ppopov1/datasets/ICA_correct_order.csv",
    tune_indices_path: str = "/data/users2/ppopov1/datasets/ukb/tune_indices.csv",
    exp_indices_path: str = "/data/users2/ppopov1/datasets/ukb/exp_indices.csv",
):
    """
    Return UKB data

    Input:
    dataset_path: str = "/data/users2/ppopov1/datasets/ukb/UKB_sex_data.npz"
    - path to the dataset with lablels
    indices_path: str = "/data/users2/ppopov1/datasets/ICA_correct_order.csv"
    - path to correct indices/components

    Output:
    features, labels
    """

    data, sexes = load_sex_data(
        cfg, 
        indices_path=indices_path, 
        tune_indices_path=tune_indices_path, 
        exp_indices_path=exp_indices_path
    )

    with np.load(dataset_path) as npzfile:
        ages = npzfile["labels"]

    if "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout:
        if cfg.mode.name == "tune":
            indices = pd.read_csv(tune_indices_path, header=None).to_numpy().squeeze()
        else:
            indices = pd.read_csv(exp_indices_path, header=None).to_numpy().squeeze()

        ages = ages[indices]

    bins = np.histogram_bin_edges(ages)
    ages = np.digitize(ages, bins)

    ages[ages == bins.shape[0]] = bins.shape[0] - 1
    ages = ages - 1

    labels = ages + sexes * np.unique(ages).shape[0]

    return data, labels
