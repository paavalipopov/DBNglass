# pylint: disable=too-many-function-args, invalid-name
""" FBIRN ICA dataset (with time direction labels) loading script"""
import numpy as np

from omegaconf import DictConfig

from src.datasets.fbirn import load_data as load_original_data


def load_data(cfg: DictConfig):
    """
    Return FBIRN normal + inversed time data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv")
    - path to labels=

    Output:
    features, labels
    """

    data, _ = load_original_data(cfg)
    inversed_data = np.flip(data, axis=1)

    labels = [0] * data.shape[0]
    labels.extend([1] * data.shape[0])
    labels = np.array(labels)

    data = np.concatenate((data, inversed_data))

    return data, labels
