# pylint: disable=too-many-function-args, invalid-name
""" HCP ICA dataset with time-direction labels loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath("hcp_time/data.npz"),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
):
    """
    Return ICA HCP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("hcp_time/HCP_AllData_sess1.npz")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("hcp_time/correct_indices_GSP.csv")
    - path to correct indices/components

    Output:
    features, labels
    """

    # get data
    with np.load(dataset_path) as npzfile:
        data = npzfile["data"]
        labels = npzfile["labels"]

    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]
        # >>> (833, 53, 1185)

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
