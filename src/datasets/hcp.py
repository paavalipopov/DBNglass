# pylint: disable=too-many-function-args, invalid-name
""" HCP ICA dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath("hcp/HCP_AllData_sess1.npz"),
    labels_path: str = DATA_ROOT.joinpath("hcp/labels_HCP_Gender.csv"),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
):
    """
    Return ICA HCP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("hcp/HCP_AllData_sess1.npz")
    - path to the dataset
    labels_path: str = DATA_ROOT.joinpath("hcp/labels_HCP_Gender.csv")
    - path to labels
    indices_path: str = DATA_ROOT.joinpath("hcp/correct_indices_GSP.csv")
    - path to correct indices/components

    Output:
    features, labels
    """

    # get data
    data = np.load(dataset_path)
    # print(data.shape)
    # >>> (833, 100, 1185)

    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]
        # >>> (833, 53, 1185)

    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int")
    # (833,)

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
