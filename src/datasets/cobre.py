# pylint: disable=too-many-function-args, invalid-name
""" COBRE ICA dataset loading script"""
import h5py
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv"),
):
    """
    Return COBRE data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("COBRE_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (157, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 157 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        # filter the data: leave only correct components
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
