# pylint: disable=too-many-function-args, invalid-name
""" OASIS ICA dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz"),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv"),
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv"),
):
    """
    Return OASIS data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv")
    - path to labels
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv")
    - path to indices of the first sessions

    Output:
    features, labels
    """

    data = np.load(dataset_path)
    # 2826 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 160 - time points - data.shape[2]

    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1

        # filter the data: leave only correct components and the first 156 time points
        # (not all subjects have all 160 time points)
        data = data[:, idx, :156]
        # print(data.shape)
        # 53 - components - data.shape[1]
        # 156 - time points - data.shape[2]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    if cfg.dataset.only_first_sessions:
        # leave only first sessions
        sessions = pd.read_csv(sessions_path, header=None)
        first_session = sessions[0].values - 1

        data = data[first_session, :, :]
        # 912 - sessions - data.shape[0] - only first session
        labels = labels[first_session]

    filter_array = []
    if cfg.dataset.multiclass:
        unique, counts = np.unique(labels, return_counts=True)
        counts = dict(zip(unique, counts))

        print(f"Number of classes in the data: {unique.shape[0]}")
        valid_labels = []
        for label, count in counts.items():
            if count > 10:
                valid_labels += [label]
            else:
                print(
                    f"There is not enough labels '{label}' in the dataset, filtering them out"
                )

        if len(valid_labels) == unique.shape[0]:
            filter_array = [True] * labels.shape[0]
        else:
            for label in labels:
                if label in valid_labels:
                    filter_array.append(True)
                else:
                    filter_array.append(False)
    else:
        # leave subjects of class 0 and 1 only
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

    data = data[filter_array, :, :]
    # 2559 - sessions - data.shape[0] - subjects of class 0 and 1
    # 823 - sessions - data.shape[0] - if only first sessions are considered
    labels = labels[filter_array]

    unique = np.sort(np.unique(labels))
    shift_dict = dict(zip(unique, np.arange(unique.shape[0])))
    for i, _ in enumerate(labels):
        labels[i] = shift_dict[labels[i]]

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
