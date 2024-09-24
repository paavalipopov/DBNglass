# pylint: disable=too-many-function-args, invalid-name
""" ABIDE ICA (869 subjects) dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath(
        "abide869/ABIDE1_AllData_869Subjects_ICA.npz"
    ),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide869/labels_ABIDE1_869Subjects.csv"),
):
    """
    Return ABIDE1 869 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv")
    - path to labels

    Output:
    features, labels
    """

    data = np.load(dataset_path)
    # 869 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 295 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1

    if cfg.dataset.filter_indices:
        # filter the data: leave only correct components and the first 156 time points
        # (not all subjects have all 160 time points)
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]
        # 156 - time points - data.shape[2]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
