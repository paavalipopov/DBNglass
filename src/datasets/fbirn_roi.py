# pylint: disable=too-many-function-args, invalid-name, unused-argument
""" FBIRN ROI dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig = None,
    dataset_path: str = DATA_ROOT.joinpath(
        "fbirn_roi/FBIRN_fMRI_200ShaeferAtlas_onlytimeserieszscored.npz"
    ),
    labels_path: str = DATA_ROOT.joinpath("fbirn_roi/labels_FBIRN_new.csv"),
):
    """
    Return ROI FBIRN data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("fbirn_roi/FBIRN_fMRI_200ShaeferAtlas_onlytimeserieszscored.npz")
    - path to the dataset
    labels_path: str = DATA_ROOT.joinpath("fbirn_roi/labels_FBIRN_new.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    data = np.load(dataset_path)
    # print(data.shape)
    # >>> (311, 200, 160)

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
