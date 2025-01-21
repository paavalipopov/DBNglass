# pylint: disable=too-many-function-args, invalid-name
""" FBIRN ICA dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig = None,
    dataset_path: str = DATA_ROOT.joinpath("fbirn/data_main.npz"),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
):
    """
    Return FBIRN data without holdout

    Output:
    data with shape [n_samples, time_length, feature_size],
    labels
    """

    # get data
    with np.load(dataset_path) as npzfile:
        data = npzfile["data"]
        labels = npzfile["diagnosis"]


    if cfg is None or cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        data = data[:, :, idx]
        # data.shape = [295, 140, 53]

    return data, labels
